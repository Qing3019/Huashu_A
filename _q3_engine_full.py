from dataclasses import dataclass
from itertools import combinations, product
import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BOX_HALF = 5000.0
BOX_MIN = np.full(3, -BOX_HALF)
BOX_MAX = np.full(3, BOX_HALF)
PERIOD = np.full(3, 2.0 * BOX_HALF)
LENGTH = 5000.0
RADIUS = 30.0
THRESHOLD = 1.8
NUM_TOL = 1e-5
FEAS_TOL = 2e-7
SUPPORT_FEAS_TOL = 1e-9
CONTACT_LIMIT = THRESHOLD + NUM_TOL
SUPPORT_STABILITY_NM = 1e-5
NEAR_THRESHOLD_TOL = 1e-4
GJK_GAP_TOL = 1e-5
GJK_FAR_GUARD = 1e-2

class DynamicUnionFind:
    def __init__(self):
        self.parent = []
        self.rank = []

    def add(self):
        index = len(self.parent)
        self.parent.append(index)
        self.rank.append(0)
        return index

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True

    def connected(self, a, b):
        return self.find(a) == self.find(b)


class Cylinder:
    def __init__(self, p, q, radius=RADIUS):
        self.p = np.asarray(p, float)
        self.q = np.asarray(q, float)
        self.radius = float(radius)
        v = self.q - self.p
        self.length = float(np.linalg.norm(v))
        if self.length <= 1e-12:
            raise ValueError("cylinder axis must have positive length")
        self.u = v / self.length
        self.center = 0.5 * (self.p + self.q)
        self.h = 0.5 * self.length
        extent = self.h * np.abs(self.u) + self.radius * np.sqrt(np.maximum(0.0, 1.0-self.u**2))
        self.aabb_min = self.center - extent
        self.aabb_max = self.center + extent

    def translated(self, shift):
        return Cylinder(self.p + shift, self.q + shift, self.radius)

    def support(self, direction):
        d = np.asarray(direction, float)
        axial = float(d @ self.u)
        radial = d - axial * self.u
        radial_norm = float(np.linalg.norm(radial))
        point = self.center + (self.h if axial >= 0 else -self.h) * self.u
        if radial_norm > 1e-14:
            point = point + self.radius * radial / radial_norm
        return point


def orthobasis(u):
    u = np.asarray(u, float)
    helper = np.eye(3)[int(np.argmin(np.abs(u)))]
    v = np.cross(u, helper)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    return np.column_stack((u, v, w))


def axis_segment_in_box(cylinder, lo, hi):
    t0, t1 = -cylinder.h, cylinder.h
    for j in range(3):
        uj = float(cylinder.u[j])
        if abs(uj) <= 1e-14:
            if not (lo[j]-1e-8 <= cylinder.center[j] <= hi[j]+1e-8):
                return None
            continue
        a = (lo[j] - cylinder.center[j]) / uj
        b = (hi[j] - cylinder.center[j]) / uj
        t0, t1 = max(t0, min(a,b)), min(t1, max(a,b))
        if t0 > t1 + 1e-8:
            return None
    if t1-t0 <= 1e-8:
        return None
    return cylinder.center+t0*cylinder.u, cylinder.center+t1*cylinder.u


def normalized_geometry(cylinder, lo, hi):
    q = orthobasis(cylinder.u)
    matrix = q @ np.diag([cylinder.h, cylinder.radius, cylinder.radius])
    scale = np.maximum(hi-lo, 1.0)
    a = np.vstack((matrix/scale[:,None], -matrix/scale[:,None]))
    b = np.r_[(hi-cylinder.center)/scale, (cylinder.center-lo)/scale]
    return matrix, a, b


def y_violation(cylinder, matrix, y, lo, hi):
    x = cylinder.center + matrix @ y
    axial = max(0.0, abs(float(y[0]))-1.0) * cylinder.h
    radial = max(0.0, float(np.linalg.norm(y[1:]))-1.0) * cylinder.radius
    box = max(0.0, float(np.max(lo-x)), float(np.max(x-hi)))
    return max(axial, radial, box)


def dykstra_start(y,a,b,cycles=20):
    """Project a y-point onto bounds, radial disk and box half-spaces."""
    projectors=[
        lambda z:np.clip(z,-1.0,1.0),
        lambda z:np.r_[z[0],z[1:]/max(1.0,float(np.linalg.norm(z[1:])))],
    ]
    for row,rhs in zip(a,b):
        denominator=float(row@row)
        projectors.append(
            lambda z,row=row,rhs=rhs,denominator=denominator:
                z-max(0.0,float(row@z-rhs))/denominator*row
        )
    point=np.asarray(y,float).copy()
    corrections=[np.zeros(3) for _ in projectors]
    for _ in range(cycles):
        for index,project in enumerate(projectors):
            shifted=point+corrections[index]
            updated=project(shifted)
            corrections[index]=shifted-updated
            point=updated
    return point


def recover_feasible_y(cylinder,matrix,a,b,y,tolerance=SUPPORT_FEAS_TOL):
    """Return a strictly feasible primal point, or None if projection stalls."""
    point=np.asarray(y,float).copy()
    if y_violation(cylinder,matrix,point,BOX_MIN,BOX_MAX)<=tolerance:
        return point
    for cycles in (20,100,500,1500):
        point=dykstra_start(point,a,b,cycles=cycles)
        if y_violation(cylinder,matrix,point,BOX_MIN,BOX_MAX)<=tolerance:
            return point
    return None


def cylinder_box_feasible(cylinder, lo, hi):
    if np.any(cylinder.aabb_max < lo-1e-9) or np.any(cylinder.aabb_min > hi+1e-9):
        return None
    segment = axis_segment_in_box(cylinder, lo, hi)
    if segment is not None:
        return 0.5*(segment[0]+segment[1])

    matrix, a, b = normalized_geometry(cylinder, lo, hi)
    closest = np.clip(cylinder.center, lo, hi)
    y0 = np.linalg.solve(matrix, closest-cylinder.center)
    y0[0] = np.clip(y0[0], -1.0, 1.0)
    radial = np.linalg.norm(y0[1:])
    if radial > 1.0:
        y0[1:] /= radial
    constraints = [
        {"type":"ineq", "fun":lambda y: b-a@y, "jac":lambda y: -a},
        {"type":"ineq", "fun":lambda y: 1-y[1]**2-y[2]**2,
         "jac":lambda y: np.array([0.0,-2*y[1],-2*y[2]])},
    ]
    starts = [y0, np.zeros(3)]
    for start in starts:
        result = minimize(
            lambda y: 0.5*float(np.dot(y-start,y-start)), start,
            jac=lambda y: y-start, method="SLSQP", bounds=[(-1,1)]*3,
            constraints=constraints, options={"ftol":1e-10,"maxiter":300},
        )
        y = result.x
        if y_violation(cylinder, matrix, y, lo, hi) <= FEAS_TOL:
            return cylinder.center + matrix@y
    return None


class Fragment:
    def __init__(self, parent_id, integer_shift, cylinder, feasible_point):
        self.parent_id = int(parent_id)
        self.integer_shift = tuple(int(x) for x in integer_shift)
        self.cylinder = cylinder
        self.center = 0.5*(np.maximum(cylinder.aabb_min,BOX_MIN)+np.minimum(cylinder.aabb_max,BOX_MAX))
        self.aabb_min = np.maximum(cylinder.aabb_min, BOX_MIN)
        self.aabb_max = np.minimum(cylinder.aabb_max, BOX_MAX)
        self.matrix, self.a, self.b = normalized_geometry(cylinder, BOX_MIN, BOX_MAX)
        initial_feasible_y=np.linalg.solve(self.matrix,feasible_point-cylinder.center)
        recovered_feasible_y=recover_feasible_y(
            cylinder,self.matrix,self.a,self.b,initial_feasible_y
        )
        self.feasible_y=(
            recovered_feasible_y if recovered_feasible_y is not None
            else initial_feasible_y
        )
        if y_violation(cylinder,self.matrix,self.feasible_y,BOX_MIN,BOX_MAX)>FEAS_TOL:
            raise RuntimeError("fragment has no numerically feasible seed point")
        self.cache = {}
        self.optimized_calls = 0
        self.solver_fallbacks = 0
        self.stability_accepts = 0
        self.dykstra_retries = 0
        self.projection_recoveries = 0

    def clear_support_cache(self):
        self.cache.clear()

    def support(self, direction):
        d = np.asarray(direction,float)
        norm = float(np.linalg.norm(d))
        if norm <= 1e-14:
            return self.cylinder.center+self.matrix@self.feasible_y
        d /= norm
        # Use the normalized direction itself as the key.  Rounding a direction
        # close to an axial sign change can incorrectly reuse the opposite cap.
        key = tuple(float(value) for value in d)
        if key in self.cache:
            return self.cache[key].copy()
        analytic = self.cylinder.support(d)
        if np.all(analytic >= BOX_MIN-SUPPORT_FEAS_TOL) and np.all(analytic <= BOX_MAX+SUPPORT_FEAS_TOL):
            self.cache[key] = analytic.copy()
            return analytic
        p = self.matrix.T@d
        objective_scale=max(float(np.linalg.norm(p)),1.0)
        p /= objective_scale
        start = np.zeros(3)
        start[0] = 1.0 if p[0]>=0 else -1.0
        rn = np.linalg.norm(p[1:])
        if rn>0: start[1:] = p[1:]/rn
        constraints = [
            {"type":"ineq","fun":lambda y:self.b-self.a@y,"jac":lambda y:-self.a},
            {"type":"ineq","fun":lambda y:1-y[1]**2-y[2]**2,
             "jac":lambda y:np.array([0.0,-2*y[1],-2*y[2]])},
        ]
        candidates=[]

        def solve_from(initial,ftol,maxiter,source):
            result=minimize(lambda y:-float(p@y),initial,jac=lambda y:-p,
                            method="SLSQP",bounds=[(-1,1)]*3,constraints=constraints,
                            options={"ftol":ftol,"maxiter":maxiter})
            recovered=recover_feasible_y(
                self.cylinder,self.matrix,self.a,self.b,result.x
            )
            if recovered is not None:
                if np.linalg.norm(recovered-result.x)>1e-12:
                    self.projection_recoveries += 1
                candidates.append((float(p@recovered),recovered,result.success,source))
            return bool(result.success and recovered is not None)

        first_success=solve_from(start,1e-10,300,"extreme")
        if not first_success:
            self.solver_fallbacks += 1
            solve_from(self.feasible_y,1e-12,1000,"feasible")
            box_start=np.linalg.solve(
                self.matrix,
                np.clip(analytic,BOX_MIN,BOX_MAX)-self.cylinder.center,
            )
            solve_from(np.clip(box_start,-1.0,1.0),1e-12,1000,"box")
            projected_start=dykstra_start(start,self.a,self.b,cycles=50)
            solve_from(projected_start,1e-12,1000,"dykstra")
            self.dykstra_retries += 1

            if len(candidates)<2:
                extra_starts=(
                    np.zeros(3),
                    -start,
                    0.5*(self.feasible_y+start),
                    0.5*(self.feasible_y+np.clip(box_start,-1.0,1.0)),
                )
                for extra_index,extra_start in enumerate(extra_starts):
                    solve_from(extra_start,1e-12,1500,f"extra_{extra_index}")
                self.dykstra_retries += len(extra_starts)

        if not candidates:
            raise RuntimeError("fragment support produced no strictly feasible primal candidate")
        best=max(candidates,key=lambda z:z[0])
        _,y,success,_=best
        if not first_success:
            other_values=sorted(
                (candidate[0] for candidate in candidates if candidate is not best),
                reverse=True,
            )
            support_spread_nm=(best[0]-other_values[0])*objective_scale if other_values else math.inf
            if support_spread_nm<=SUPPORT_STABILITY_NM:
                self.stability_accepts += 1
                success=True
        self.optimized_calls += 1
        if not success:
            raise RuntimeError(
                "fragment support remained unresolved after retry; "
                f"strict objectives/status={[(value, ok, source) for value, _, ok, source in candidates]}"
            )
        point=self.cylinder.center+self.matrix@y
        if y_violation(self.cylinder,self.matrix,y,BOX_MIN,BOX_MAX)>SUPPORT_FEAS_TOL:
            raise RuntimeError("support point lost strict primal feasibility")
        self.cache[key]=point.copy()
        return point


def fragment_shift_ranges(cylinder):
    lower=np.ceil((cylinder.aabb_min-BOX_MAX)/PERIOD-1e-12).astype(int)
    upper=np.floor((cylinder.aabb_max-BOX_MIN)/PERIOD+1e-12).astype(int)
    return [range(int(lower[j]),int(upper[j])+1) for j in range(3)]


def generate_parent_fragments(parent_id, cylinder):
    fragments=[]
    for k in product(*fragment_shift_ranges(cylinder)):
        k=np.asarray(k,int)
        shifted=cylinder.translated(-k*PERIOD)
        feasible=cylinder_box_feasible(shifted,BOX_MIN,BOX_MAX)
        if feasible is not None:
            fragments.append(Fragment(parent_id,k,shifted,feasible))
    return fragments

@dataclass
class SupportPoint:
    w: np.ndarray
    a: np.ndarray
    b: np.ndarray


@dataclass
class GJKResult:
    lower: float
    upper: float
    a: np.ndarray
    b: np.ndarray
    status: str


def closest_simplex(simplex):
    points=np.asarray([s.w for s in simplex])
    best=None
    for size in range(1,min(4,len(simplex))+1):
        for ids in combinations(range(len(simplex)),size):
            ids=np.asarray(ids,int); face=points[ids]
            gram=face@face.T
            kkt=np.block([[gram,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
            lam=np.linalg.lstsq(kkt,np.r_[np.zeros(size),1.0],rcond=None)[0][:size]
            if np.min(lam)<-1e-9: continue
            lam=np.maximum(lam,0); lam/=lam.sum()
            close=lam@face; value=float(close@close)
            if best is None or value<best[0]: best=(value,ids,lam,close)
    _,ids,lam,close=best
    pa=sum(lam[t]*simplex[idx].a for t,idx in enumerate(ids))
    pb=sum(lam[t]*simplex[idx].b for t,idx in enumerate(ids))
    return close,ids,pa,pb


def gjk(shape_a,shape_b,threshold=THRESHOLD,allow_lower=True,maxiter=100):
    def supp(d):
        a=shape_a.support(d); b=shape_b.support(-np.asarray(d,float))
        return SupportPoint(a-b,a,b)
    direction=shape_b.center-shape_a.center
    if np.linalg.norm(direction)<1e-14: direction=np.array([1.,0,0])
    simplex=[supp(direction)]; lower=0.0; pa=simplex[0].a; pb=simplex[0].b
    for _ in range(maxiter):
        close,active,pa,pb=closest_simplex(simplex)
        simplex=[simplex[i] for i in active]
        upper=float(np.linalg.norm(close))
        if upper<=1e-10: return GJKResult(0,0,pa,pb,"intersect")
        if threshold is not None and upper<=threshold+NUM_TOL: return GJKResult(lower,upper,pa,pb,"connected")
        new=supp(-close)
        lower=max(lower,max(0.0,float(close@new.w/upper)))
        if threshold is not None and allow_lower and lower>threshold+NUM_TOL:
            return GJKResult(lower,upper,pa,pb,"separated")
        duplicate=any(np.linalg.norm(new.w-old.w)<=1e-9 for old in simplex)
        if duplicate or upper-lower<=1e-9*max(1.,upper):
            return GJKResult(lower,upper,pa,pb,"converged")
        simplex.append(new)
    return GJKResult(lower,float(np.linalg.norm(pa-pb)),pa,pb,"maxiter")


def segment_distances_one_to_many(p,q,p2,q2):
    d1=q-p
    d2=q2-p2
    r=p[None,:]-p2
    a=float(d1@d1)
    e=np.einsum('ij,ij->i',d2,d2)
    b=d2@d1
    c=r@d1
    f=np.einsum('ij,ij->i',d2,r)
    denom=a*e-b*b
    s=np.zeros_like(denom)
    np.divide(b*f-c*e,denom,out=s,where=denom>1e-12)
    s=np.clip(s,0,1)
    t=(b*s+f)/e
    low=t<0; high=t>1
    t=np.clip(t,0,1)
    s=np.where(low,np.clip(-c/a,0,1),s)
    s=np.where(high,np.clip((b-c)/a,0,1),s)
    cp1=p[None,:]+s[:,None]*d1
    cp2=p2+t[:,None]*d2
    dist=np.linalg.norm(cp1-cp2,axis=1)
    return dist,s,t,cp1,cp2


def in_box(point):
    return bool(np.all(point>=BOX_MIN-2e-7) and np.all(point<=BOX_MAX+2e-7))


def fragments_connect(a,b,axis_distance,s,t,cp_a,cp_b,stats):
    # 最常见的侧面-侧面情形可直接构造盒内见证点。
    if 1e-9<s<1-1e-9 and 1e-9<t<1-1e-9:
        v=cp_b-cp_a; d=float(axis_distance)
        if d<=2*RADIUS:
            wa=wb=0.5*(cp_a+cp_b)
        else:
            wa=cp_a+RADIUS*v/d
            wb=cp_b-RADIUS*v/d
        if in_box(wa) and in_box(wb):
            if float(np.linalg.norm(wa-wb))<=CONTACT_LIMIT:
                stats['side_witness']+=1
                return True
    stats['full_gjk']+=1
    full=gjk(a.cylinder,b.cylinder,allow_lower=True)
    if full.lower>CONTACT_LIMIT:
        return False
    if full.upper<=CONTACT_LIMIT and in_box(full.a) and in_box(full.b):
        return True
    stats['clipped_gjk']+=1
    clipped=gjk(a,b,allow_lower=False,maxiter=180)
    if clipped.status=="maxiter":
        raise RuntimeError("clipped GJK did not converge")
    if clipped.upper<=CONTACT_LIMIT:
        return True
    gap=abs(clipped.upper-clipped.lower)
    separated=(
        clipped.lower>CONTACT_LIMIT
        and (gap<=GJK_GAP_TOL or min(clipped.lower,clipped.upper)>CONTACT_LIMIT+GJK_FAR_GUARD)
    )
    if abs(clipped.upper-THRESHOLD)<=NEAR_THRESHOLD_TOL or not separated:
        a.clear_support_cache(); b.clear_support_cache()
        clipped=gjk(a,b,allow_lower=False,maxiter=500)
        if clipped.status=="maxiter":
            raise RuntimeError("near-threshold clipped GJK did not converge")
        stats['near_threshold']+=1
        if clipped.upper<=CONTACT_LIMIT:
            return True
        gap=abs(clipped.upper-clipped.lower)
        separated=(
            clipped.lower>CONTACT_LIMIT
            and (gap<=GJK_GAP_TOL or min(clipped.lower,clipped.upper)>CONTACT_LIMIT+GJK_FAR_GUARD)
        )
        if not separated:
            raise RuntimeError(
                "clipped GJK remained numerically unresolved: "
                f"lower={clipped.lower}, upper={clipped.upper}"
            )
    return False

def sample_cylinders(n,rng):
    centers=rng.uniform(-BOX_HALF,BOX_HALF,size=(n,3))
    directions=rng.normal(size=(n,3))
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    p=centers-0.5*LENGTH*directions
    q=centers+0.5*LENGTH*directions
    return [Cylinder(p[i],q[i]) for i in range(n)]


def simulate_nested(seed,n_levels=np.array([354,424,495,707]),base_seed=20260808):
    seed_sequence=np.random.SeedSequence([int(base_seed),int(seed)])
    rng=np.random.Generator(np.random.PCG64(seed_sequence))
    cylinders=sample_cylinders(int(n_levels[-1]),rng)
    dsu=DynamicUnionFind(); left=dsu.add(); right=dsu.add()
    capacity=8*int(n_levels[-1])
    mins=np.empty((capacity,3)); maxs=np.empty((capacity,3))
    ps=np.empty((capacity,3)); qs=np.empty((capacity,3))
    fragments=[]; indicators=np.zeros(len(n_levels),dtype=bool)
    stats={'fragments':0,'aabb_candidates':0,'axis_candidates':0,'side_witness':0,
           'full_gjk':0,'clipped_gjk':0,'support_calls':0,'fallbacks':0,
           'dykstra_retries':0,'projection_recoveries':0,
           'stability_accepts':0,'near_threshold':0}
    level=0
    left_slab_min=BOX_MIN.copy(); left_slab_max=BOX_MAX.copy(); left_slab_max[0]=BOX_MIN[0]+CONTACT_LIMIT
    right_slab_min=BOX_MIN.copy(); right_slab_min[0]=BOX_MAX[0]-CONTACT_LIMIT; right_slab_max=BOX_MAX.copy()

    for parent_id,cylinder in enumerate(cylinders):
        for frag in generate_parent_fragments(parent_id,cylinder):
            node=dsu.add(); m=len(fragments)
            if frag.aabb_min[0]<=left_slab_max[0]+1e-10 and cylinder_box_feasible(frag.cylinder,left_slab_min,left_slab_max) is not None:
                dsu.union(left,node)
            if frag.aabb_max[0]>=right_slab_min[0]-1e-10 and cylinder_box_feasible(frag.cylinder,right_slab_min,right_slab_max) is not None:
                dsu.union(right,node)
            if m:
                gap=np.maximum(0.0,np.maximum(frag.aabb_min-maxs[:m],mins[:m]-frag.aabb_max))
                ids=np.flatnonzero(np.einsum('ij,ij->i',gap,gap)<=CONTACT_LIMIT**2+1e-12)
                stats['aabb_candidates']+=len(ids)
                if len(ids):
                    dist,s,t,cpa,cpb=segment_distances_one_to_many(frag.cylinder.p,frag.cylinder.q,ps[ids],qs[ids])
                    keep=np.flatnonzero(dist<=2*RADIUS+CONTACT_LIMIT+1e-10)
                    stats['axis_candidates']+=len(keep)
                    for pos in keep:
                        old_id=int(ids[pos])
                        if dsu.connected(node,old_id+2):
                            continue
                        if fragments_connect(frag,fragments[old_id],dist[pos],s[pos],t[pos],cpa[pos],cpb[pos],stats):
                            dsu.union(node,old_id+2)
            fragments.append(frag)
            mins[m]=frag.aabb_min; maxs[m]=frag.aabb_max
            ps[m]=frag.cylinder.p; qs[m]=frag.cylinder.q

        count=parent_id+1
        if dsu.connected(left,right):
            indicators[level:]=True
            stats['fragments']=len(fragments)
            stats['support_calls']=sum(f.optimized_calls for f in fragments)
            stats['fallbacks']=sum(f.solver_fallbacks for f in fragments)
            stats['dykstra_retries']=sum(f.dykstra_retries for f in fragments)
            stats['projection_recoveries']=sum(f.projection_recoveries for f in fragments)
            stats['stability_accepts']=sum(f.stability_accepts for f in fragments)
            return indicators,stats
        while level<len(n_levels) and count==n_levels[level]:
            indicators[level]=dsu.connected(left,right)
            level+=1
    stats['fragments']=len(fragments)
    stats['support_calls']=sum(f.optimized_calls for f in fragments)
    stats['fallbacks']=sum(f.solver_fallbacks for f in fragments)
    stats['dykstra_retries']=sum(f.dykstra_retries for f in fragments)
    stats['projection_recoveries']=sum(f.projection_recoveries for f in fragments)
    stats['stability_accepts']=sum(f.stability_accepts for f in fragments)
    return indicators,stats


def simulate_nested_checked(seed,n_levels=np.array([354,424,495,707]),base_seed=20260808):
    try:
        return simulate_nested(seed,n_levels,base_seed)
    except Exception as exc:
        raise RuntimeError(f"Monte Carlo trial {seed} failed") from exc


def run_monte_carlo(trial_count,n_levels=np.array([354,424,495,707]),base_seed=20260808,n_jobs=1,trial_start=0):
    """固定试验数；每个试验用独立、可复现的(seed, trial_index)随机流。"""
    trial_indices=list(range(int(trial_start),int(trial_start)+int(trial_count)))
    started=time.perf_counter()
    if n_jobs==1:
        outputs=[]
        report_every=max(1,int(trial_count)//10)
        for trial_index in trial_indices:
            try:
                outputs.append(simulate_nested_checked(trial_index,n_levels,base_seed))
            except Exception as exc:
                raise RuntimeError(f"蒙特卡洛第{trial_index}次试验失败") from exc
            local_count=trial_index-int(trial_start)+1
            if local_count%report_every==0 or local_count==trial_count:
                print(f"完成 {local_count}/{trial_count} 次试验")
    else:
        from joblib import Parallel,delayed
        outputs=Parallel(n_jobs=n_jobs,backend="loky")(
            delayed(simulate_nested_checked)(i,n_levels,base_seed) for i in trial_indices
        )
    indicators=np.asarray([item[0] for item in outputs],dtype=bool)
    diagnostics=pd.DataFrame([item[1] for item in outputs])
    assert np.all(indicators[:,:-1]<=indicators[:,1:]), "共同前缀的导通结果违反单调性。"
    return indicators,diagnostics,time.perf_counter()-started


def wilson_interval(successes,trials,z=1.959963984540054):
    successes=np.asarray(successes,float)
    n=float(trials)
    p=successes/n
    denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return np.maximum(0,center-radius),np.minimum(1,center+radius)

SEARCH_LEVELS = np.arange(1, 708, dtype=int)
NO_HIT_SENTINEL = int(SEARCH_LEVELS[-1]) + 1


def simulate_first_passage(seed, max_cylinders=707, base_seed=20260808):
    """Return the first complete parent-cylinder count that connects the box.

    This is intentionally a thin wrapper around the Q2 nested-prefix engine.
    The sentinel ``max_cylinders + 1`` means that no connection was observed
    within the simulated upper bound; it is not an observed connection count.
    """
    max_cylinders = int(max_cylinders)
    levels = (
        SEARCH_LEVELS
        if max_cylinders == int(SEARCH_LEVELS[-1])
        else np.arange(1, max_cylinders + 1, dtype=int)
    )
    indicators, original_stats = simulate_nested_checked(
        int(seed), levels, int(base_seed)
    )
    hits = np.flatnonzero(indicators)
    first_passage = int(hits[0] + 1) if len(hits) else max_cylinders + 1
    reconstructed = levels >= first_passage
    if not np.array_equal(reconstructed, indicators):
        raise AssertionError("first-passage reconstruction violated monotonicity")
    stats = {"activated_cylinders": min(first_passage, max_cylinders)}
    stats.update(original_stats)
    return first_passage, stats


def simulate_first_passage_checked(seed, max_cylinders=707, base_seed=20260808):
    try:
        return simulate_first_passage(seed, max_cylinders, base_seed)
    except Exception as exc:
        raise RuntimeError(
            f"Monte Carlo first-passage trial {seed} failed"
        ) from exc


def run_first_passage_monte_carlo(
    trial_count,
    max_cylinders=707,
    base_seed=20260808,
    n_jobs=1,
    trial_start=0,
):
    """Run reproducible independent trials and return first-passage counts."""
    trial_indices = list(
        range(int(trial_start), int(trial_start) + int(trial_count))
    )
    started = time.perf_counter()
    if int(n_jobs) == 1:
        outputs = []
        report_every = max(1, int(trial_count) // 10)
        for local_index, trial_index in enumerate(trial_indices, start=1):
            outputs.append(
                simulate_first_passage_checked(
                    trial_index, max_cylinders, base_seed
                )
            )
            if local_index % report_every == 0 or local_index == trial_count:
                print(f"completed {local_index}/{trial_count} trials")
    else:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=int(n_jobs), backend="loky")(
            delayed(simulate_first_passage_checked)(
                trial_index, max_cylinders, base_seed
            )
            for trial_index in trial_indices
        )

    first_passage = np.asarray([item[0] for item in outputs], dtype=np.int16)
    diagnostics = pd.DataFrame([item[1] for item in outputs])
    sentinel = int(max_cylinders) + 1
    if np.any((first_passage < 1) | (first_passage > sentinel)):
        raise AssertionError("invalid first-passage count")
    return first_passage, diagnostics, time.perf_counter() - started


def one_sided_wilson(successes, trials, z=1.6448536269514722):
    """Return marginal one-sided Wilson lower and upper bounds."""
    return wilson_interval(successes, trials, z=z)
