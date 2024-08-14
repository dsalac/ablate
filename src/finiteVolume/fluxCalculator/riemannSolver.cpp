
#include "riemannSolver.hpp"
#include "utilities/constants.hpp"

using namespace ablate::finiteVolume::fluxCalculator;

static void expansionShockCalculation(const PetscReal pstar, const PetscReal gamma, const PetscReal gamm1, const PetscReal gamp1, const PetscReal p0, const PetscReal p, const PetscReal a,
                                      const PetscReal rho, PetscReal *f0, PetscReal *f1) {
    if (pstar <= p)  // expansion wave equation from Toro
    {
        PetscReal A = 2.0 * a / gamm1;
        PetscReal B = 0.5 * gamm1 / gamma;
        PetscReal pRatio = (pstar + p0) / (p + p0);
        PetscReal pPow = PetscPowReal(pRatio, B);

        *f0 = A * (pPow - 1.0);
        *f1 = A * B * pPow / (pstar + p0);

    } else  // shock equation from Toro
    {
        PetscReal A = 2 / (gamp1 * rho);
        PetscReal B = gamm1 * (p + p0) / gamp1;
        PetscReal pFrac = A / (pstar + p0 + B);
        PetscReal pSqrt = PetscSqrtReal(pFrac);

        *f0 = (pstar - p) * pSqrt;
        *f1 = 0.5 * pFrac * pSqrt * (2.0 * (B + p0) + pstar + p) / A;
    }
}

static Direction riemannDirection(const PetscReal pstar, const PetscReal uL, const PetscReal aL, const PetscReal rhoL, const PetscReal p0L, const PetscReal pL, const PetscReal gammaL,
                                  const PetscReal fL, const PetscReal uR, const PetscReal aR, const PetscReal rhoR, const PetscReal p0R, const PetscReal pR, const PetscReal gammaR, const PetscReal fR,
                                  PetscReal *massFlux, PetscReal *p12) {
    /*
     * gammaL: specific heat ratio for gas on left (pass in from EOS)
     * gammaR: specific heat ratio for stiffened gas on right (pass in from EOS)
     * p0L: reference pressure for stiffened gas on left (pass in from EOS)
     * p0R: reference pressure for stiffened gas on right (pass in from EOS)
     * uL: velocity on the left cell center
     * uR: velocity on the right cell center
     * rhoR: density on the right cell center
     * rhoL: density on the left cell center
     * pR: pressure on the right cell center
     * pL: pressure on the left cell center
     * aR: SoS on the right center cell
     * aL: SoS on the left center cell
     * pstar: pressure across contact surface
     * ustar: velocity across contact surface
     */

    PetscReal STLR, SHLR, A, pRatio, gamma, gamm1, gamp1, astar, uX;

    // Now, start backing out the rest of the info.
    PetscReal ustar = 0.5 * (uL + uR + fR - fL);
    if (ustar >= 0) {
        gamma = gammaL;
        gamm1 = gamma - 1.0;
        gamp1 = gamma + 1.0;
        if (pstar <= pL)  // left expansion
        {
            astar = aL * PetscPowReal((pstar + p0L) / (pL + p0L), (gamm1 / (2 * gamma)));
            STLR = ustar - astar;
            if (STLR >= 0)  // positive tail wave
            {
                SHLR = uL - aL;
                if (SHLR >= 0)  // positive head wave
                {
                    *massFlux = rhoL * uL;
                    *p12 = pL;
                    uX = uL;
                } else  // Eq. 4.56 negative head wave
                {
                    A = rhoL * PetscPowReal((2 / gamp1) + ((gamm1 * uL) / (gamp1 * aL)), (2 / gamm1));
                    uX = 2 / gamp1 * (aL + (gamm1 * uL / 2));
                    *massFlux = A * uX;
                    *p12 = (pL + p0L) * PetscPowReal((2 / gamp1) + ((gamm1 * uL) / (gamp1 * aL)), (2 * gamma / gamm1));
                }
            } else {
                pRatio = (pstar + p0L) / (pL + p0L);
                *massFlux = rhoL * PetscPowReal(pRatio, 1.0 / gamma) * ustar;
                *p12 = pstar;
                uX = ustar;
            }

        } else  // Left shock
        {
            A = sqrt((gamp1 * (pstar + p0L) / 2 / gamma / (pL + p0L)) + (gamm1 / 2 / gamma));
            STLR = uL - (aL * A);  // shock wave speed
            if (STLR >= 0) {
                *massFlux = rhoL * uL;
                *p12 = pL;
                uX = uL;
            } else  // negative wave speed
            {
                pRatio = (pstar + p0L) / (pL + p0L);
                *massFlux = rhoL * (pRatio + (gamm1 / gamp1)) / (gamm1 * pRatio / gamp1 + 1) * ustar;
                *p12 = pstar;
                uX = ustar;
            }
        }
    } else  // negative ustar
    {
        gamma = gammaR;
        gamm1 = gamma - 1.0;
        gamp1 = gamma + 1.0;
        if (pstar <= pR)  // right expansion
        {
            SHLR = uR + aR;
            if (SHLR >= 0)  // positive head wave
            {
                astar = aR * PetscPowReal((pstar + p0R) / (pR + p0R), (gamm1 / (2 * gamma)));
                STLR = ustar + astar;
                if (STLR >= 0)  // positive tail wave
                {
                    pRatio = (pstar + p0R) / (pR + p0R);
                    *massFlux = rhoR * PetscPowReal(pRatio, 1 / gamma) * ustar;
                    *p12 = pstar;
                    uX = ustar;
                } else  // Eq. 4.56 negative tail wave
                {
                    A = rhoR * PetscPowReal((2 / gamp1) - ((gamm1 * uR) / (gamp1 * aR)), (2 / gamm1));
                    uX = 2 / gamp1 * (-aR + (gamm1 * uR / 2));
                    *massFlux = A * uX;
                    *p12 = (pR + p0R) * PetscPowReal((2 / gamp1) - ((gamm1 * uR) / (gamp1 * aR)), (2 * gamma / gamm1));
                }
            } else  // negative head wave
            {
                *massFlux = rhoR * uR;
                *p12 = pR;
                uX = uR;
            }
        } else  // right shock
        {
            A = sqrt((gamp1 * (pstar + p0R) / 2 / gamma / (pR + p0R)) + (gamm1 / 2 / gamma));
            STLR = uR + (aR * A);  // shock wave speed
            if (STLR >= 0) {
                pRatio = (pstar + p0R) / (pR + p0R);
                *massFlux = rhoR * (pRatio + (gamm1 / gamp1)) / (gamm1 * pRatio / gamp1 + 1) * ustar;
                *p12 = pstar;
                uX = ustar;
            } else  // negative wave speed
            {
                *massFlux = rhoR * uR;
                *p12 = pR;
                uX = uR;
            }
        }
    }

    return (uX > 0 ? Direction::LEFT : Direction::RIGHT);
}

//static PetscInt cnt = 0;

// Solve the non-linear equation
Direction RiemannSolver::riemannSolver(const PetscReal uL, const PetscReal aL, const PetscReal rhoL, const PetscReal p0L, const PetscReal pL, const PetscReal gammaL, const PetscReal uR,
                                       const PetscReal aR, const PetscReal rhoR, const PetscReal p0R, const PetscReal pR, const PetscReal gammaR, const PetscReal pstar0, PetscReal *massFlux,
                                       PetscReal *p12) {
    const PetscReal tol = 1e-8;
    PetscReal pold, f_L_0, f_L_1, f_R_0, f_R_1, pstar = pstar0;
    const PetscReal del_u = uR - uL;
    const PetscReal gamLm1 = gammaL - 1.0, gamLp1 = gammaL + 1.0;
    const PetscReal gamRm1 = gammaR - 1.0, gamRp1 = gammaR + 1.0;
    const PetscInt MAXIT = 100;
    PetscInt i = 0;
//++cnt;

//if (cnt==28446) {
//  FILE *f1 = fopen("pstar.txt", "w");
//  printf("%+e\n", pstar0);
//  for (PetscReal p = 0; p <= 5; p += 5/1e6) {
//    expansionShockCalculation(p, gammaL, gamLm1, gamLp1, p0L, pL, aL, rhoL, &f_L_0, &f_L_1);
//    expansionShockCalculation(p, gammaR, gamRm1, gamRp1, p0R, pR, aR, rhoR, &f_R_0, &f_R_1);
//    fprintf(f1, "%+e\t%+e\n", p, f_L_0 + f_R_0 + del_u);
//  }
//  fclose(f1);
////  exit(0);
//}

    do  // Newton's method
    {
        expansionShockCalculation(pstar, gammaL, gamLm1, gamLp1, p0L, pL, aL, rhoL, &f_L_0, &f_L_1);
        expansionShockCalculation(pstar, gammaR, gamRm1, gamRp1, p0R, pR, aR, rhoR, &f_R_0, &f_R_1);

        pold = pstar;
        pstar = pold - 0.5*(f_L_0 + f_R_0 + del_u) / (f_L_1 + f_R_1);  // new guess


        // A stiffened gas will have p0L and p0R as positive numbers. If they're both zero (or close enough) then don't allow
        //  for a negative pstar. Set the value to something just above zero.
        if (pstar < 0 && (p0L < ablate::utilities::Constants::tiny || p0R < ablate::utilities::Constants::tiny)) {
            pstar = ablate::utilities::Constants::small;
        }

        i++;

//        if (cnt==28446) printf("%ld: %+e\t%e\t%+e\n", i, pstar, PetscAbsReal((pstar - pold) / pstar),f_L_0 + f_R_0 + del_u);
    } while (PetscAbsReal((pstar - pold) / pstar) > tol && i <= MAXIT);

    if (i > MAXIT) {
      PetscReal a = 0.0, b = pstar0, m;
      PetscReal fa, fb;

      expansionShockCalculation(a, gammaL, gamLm1, gamLp1, p0L, pL, aL, rhoL, &f_L_0, &f_L_1);
      expansionShockCalculation(a, gammaR, gamRm1, gamRp1, p0R, pR, aR, rhoR, &f_R_0, &f_R_1);
      fa = f_L_0 + f_R_0 + del_u;

      expansionShockCalculation(b, gammaL, gamLm1, gamLp1, p0L, pL, aL, rhoL, &f_L_0, &f_L_1);
      expansionShockCalculation(b, gammaR, gamRm1, gamRp1, p0R, pR, aR, rhoR, &f_R_0, &f_R_1);
      fb = f_L_0 + f_R_0 + del_u;

      if (fa*fb>0.0) {
        FILE *f1 = fopen("pstar.txt", "w");
        printf("%+e\n", pstar0);
        for (PetscReal p = 0; p <= pstar0; p += pstar0/1e7) {
          expansionShockCalculation(p, gammaL, gamLm1, gamLp1, p0L, pL, aL, rhoL, &f_L_0, &f_L_1);
          expansionShockCalculation(p, gammaR, gamRm1, gamRp1, p0R, pR, aR, rhoR, &f_R_0, &f_R_1);
          fprintf(f1, "%+e\t%+e\n", p, f_L_0 + f_R_0 + del_u);
        }
        fclose(f1);

        throw std::runtime_error("Can't find pstar; No valid bracket. "+std::to_string(fa)+" "+std::to_string(fb));
      }

      i = 0;
      do
      {
        m = (b*fa - a*fb)/(fa - fb);
        expansionShockCalculation(m, gammaL, gamLm1, gamLp1, p0L, pL, aL, rhoL, &f_L_0, &f_L_1);
        expansionShockCalculation(m, gammaR, gamRm1, gamRp1, p0R, pR, aR, rhoR, &f_R_0, &f_R_1);
        PetscReal fm = f_L_0 + f_R_0 + del_u;

        if (fa*fm<=0.0) {
          b = m;
          fb = fm;
        }
        else {
          a = m;
          fa = fm;
        }

//        printf("%ld: %+e\t%+e\t%e\t%+e\n", i, a, b, (b-a)/m, fm);
        i++;

      } while ((b - a)/m > tol && i <= MAXIT);
//exit(0);
      if (i > MAXIT) {
        throw std::runtime_error("Can't find pstar; Iteration not converging; Go back and do it again");
//        throw std::runtime_error("Can't find pstar; Iteration not converging; Go back and do it again " + std::to_string(cnt));
      }

      pstar = m;


    }


    return riemannDirection(pstar, uL, aL, rhoL, p0L, pL, gammaL, f_L_0, uR, aR, rhoR, p0R, pR, gammaR, f_R_0, massFlux, p12);
}
