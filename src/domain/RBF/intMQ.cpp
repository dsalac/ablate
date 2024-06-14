#include "intMQ.hpp"

using namespace ablate::domain::rbf;

/************ Begin Multiquadric Derived Class **********************/

intMQ::intMQ(int p, double scale, bool doesNotHaveDerivatives, bool doesNotHaveInterpolation, bool returnNeighborVertices)
    : RBF(p, !doesNotHaveDerivatives, !doesNotHaveInterpolation, returnNeighborVertices), scale(scale < PETSC_SMALL ? __RBF_intMQ_DEFAULT_PARAM : scale){

      a[1] = scale;
      a[2] = PetscSqr(a[1]);
      a[3] = a[2]*a[1];
      a[4] = PetscSqr(a[2]);

};

// Values and derivatives of the RBF as a function of radius
#define p0(a, r, mq) ((a[4]/45.0 - 83.0*a[2]*r[2]/720.0 + r[4]/120)*mq + (-a[4]*r[1]/16.0 + a[2]*r[3]/12.0)*PetscLogReal((r[1] + mq)/a[1]))
#define p1(a, r, mq) ((-13.0*a[2]*r[1]/48.0 + r[3]/24.0)*mq + (a[2]*r[2]/4.0 - a[4]/16.0)*PetscLogReal((r[1] + mq)/a[1]))
#define p2(a, r, mq) ((r[2]/6.0 - a[2]/3.0)*mq + 0.5*a[2]*r[1]*PetscLogReal((r[1] + mq)/a[1]))
#define p3(a, r, mq) (0.5*(r[1]*mq + a[2]*PetscLogReal((r[1] + mq)/a[1])))

//#define p0(a, r, mq) ((mq*(16.0*a[4] - 83.0*a[2]*r[2] + 6.0*r[4]) + (60.0*a[1]*r[3] - 45.0*a[3]*r[1])*PetscAsinhReal(r[1]/a[1]))/720.0)
//#define p1(a, r, mq) ((mq*r[1]*(2.0*r[2] - 13.0*a[2]) - 3.0*(a[3] - 4.0*a[1]*r[2])*PetscAsinhReal(r[1]/a[1]))/48.0)
//#define p2(a, r, mq) ((mq*(r[2] - 2.0*a[2]) + 3.0*a[1]*r[1]*PetscAsinhReal(r[1]/a[1]))/6.0)
//#define p3(a, r, mq) (0.5*(mq*r[1] + a[1]*PetscAsinReal(r[1]/a[1])))


//#define p0(a, r, mq) (mq)
//#define p1(a, r, mq) (r[1]/(a[2]*mq))
//#define p2(a, r, mq) (1.0/((a[2]+r[2])*mq))
//#define p3(a, r, mq) (-3.0*r[1]/(PetscSqr(a[2]+r[2])*mq))I

//#define p0(a, r, mq) (a[1]*(2.0*a[1]*mq*(a[2] + r[2]) + PetscSqrtReal(PETSC_PI)*r[1]*(3.0*a[2]+2.0*r[2])*PetscErfReal(r[1]/a[1]))/24.0)
//#define p1(a, r, mq) (a[1]*(2.0*a[1]*mq*r[1] + PetscSqrtReal(PETSC_PI)*(a[2]+2.0*r[2])*PetscErfReal(r[1]/a[1]))/8.0)
//#define p2(a, r, mq) (0.5*a[1]*(a[1]*mq+PetscSqrtReal(PETSC_PI)*r[1]*PetscErfReal(r[1]/a[1])))
//#define p3(a, r, mq) (0.5*a[1]*PetscSqrtReal(PETSC_PI)*PetscErfReal(r[1]/a[1]))

// Multiquadric: sqrt(1+(er)^2)
PetscReal intMQ::RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) {
    PetscReal r[5] = {1.0, 0.0, 0.0, 0.0, 0.0};
    r[2] = intMQ::DistanceSquared(dim, x, y);
    r[1] = PetscSqrtReal(r[2]);
    r[3] = r[2]*r[1];
    r[4] = PetscSqr(r[2]);
    const PetscReal mq = (PetscSqrtReal(r[2] + a[2]));
//    const PetscReal mq = (PetscSqrtReal(1 + r[2]/a[2]));
//    const PetscReal mq = PetscExpReal(-r[2]/a[2]);

    return p0(a, r, mq);
}

// Derivatives of the radius with respect to directions
#define r1(x, r) (x/r[1])
#define r2(x, r) ((r[2] - x*x)/r[3])
#define rCross(x1, x2, r) ((-x1*x2)/r[3])
#define rCross3(x, r) ((x[0]*x[1]*x[2])/(r[2]*r[3]))

// Apply the limit as r->0
#define LimitFcn(r, val, lm) (r[1] > PETSC_MACHINE_EPSILON ? val : lm)


// Derivatives of integrated Multiquadric spline at a location.
PetscReal intMQ::RBFDer(PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {
    PetscReal r[5] = {1.0, 0.0, 0.0, 0.0, 0.0};
    r[2] = intMQ::DistanceSquared(dim, x);
    r[1] = PetscSqrtReal(r[2]);
    r[3] = r[2]*r[1];
    r[4] = PetscSqr(r[2]);
    const PetscReal mq = PetscSqrtReal(r[2] + a[2]);
//    const PetscReal mq = (PetscSqrtReal(1 + r[2]/a[2]));
//    const PetscReal mq = PetscExpReal(-r[2]/a[2]);

    switch (dx + 10 * dy + 100 * dz) {
        case 0:
            return  p0(a, r, mq);
        case 1:  // x
            return LimitFcn(r, (p1(a, r, mq)*r1(x[0], r)), 0.0);
        case 10:  // y
            return LimitFcn(r, (p1(a, r, mq)*r1(x[1], r)), 0.0);
        case 100:  // z
            return LimitFcn(r, (p1(a, r, mq)*r1(x[2], r)), 0.0);
        case 2:  // xx
            return LimitFcn(r, (p2(a, r, mq)*PetscSqr(r1(x[0], r)) + p1(a, r, mq)*r2(x[0], r)), a[2]/2.0);
        case 20:  // yy
            return LimitFcn(r, (p2(a, r, mq)*PetscSqr(r1(x[1], r)) + p1(a, r, mq)*r2(x[1], r)), a[2]/2.0);
        case 200:  // zz
            return LimitFcn(r, (p2(a, r, mq)*PetscSqr(r1(x[2], r)) + p1(a, r, mq)*r2(x[2], r)), a[2]/2.0);
        case 11:  // xy
            return LimitFcn(r, (p2(a, r, mq)*r1(x[0], r)*r1(x[1], r) + p1(a, r, mq)*rCross(x[0], x[1], r)), 0.0);
        case 101:  // xz
            return LimitFcn(r, (p2(a, r, mq)*r1(x[0], r)*r1(x[2], r) + p1(a, r, mq)*rCross(x[0], x[2], r)), 0.0);
        case 110:  // yz
            return LimitFcn(r, (p2(a, r, mq)*r1(x[1], r)*r1(x[2], r) + p1(a, r, mq)*rCross(x[1], x[2], r)), 0.0);
        case 111:  // xyz
        {
            return LimitFcn(r, (p3(a, r, mq)*r1(x[0], r)*r1(x[1], r)*r1(x[2], r)
                      + p2(x, r, mq)*(r1(x[0], r)*rCross(x[1], x[2], r) + r1(x[1], r)*rCross(x[0], x[2], r) + r1(x[2], r)*rCross(x[0], x[1], r))
                      + p1(x, r, mq)*rCross3(x, r)), 0.0);
        }
        default:
            throw std::invalid_argument("intMQ: Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
    }

    return 0.0;
}

/************ End Multiquadric Derived Class **********************/

#include "registrar.hpp"
REGISTER(ablate::domain::rbf::RBF, ablate::domain::rbf::intMQ, "Integrated MQ Radial Basis Function",
         OPT(int, "polyOrder", "Order of the augmenting RBF polynomial. Must be >= 1. Any value <1 will result in a polyOrder of 4."),
         OPT(double, "scale", "Scaling parameter. Must be >0. Any value <PETSC_SMALL will result in a default scale of 0.1."),
         OPT(bool, "doesNotHaveDerivatives", "Compute derivative information. Default is false."),
         OPT(bool, "doesNotHaveInterpolation", "Compute interpolation information. Default is false."),
         OPT(bool, "useNeighborVertices", "Perform RBF based on neighboring vertices (TRUE) or cells (FALSE). Default is false."));
