#include "nPhaseIntSharp.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include "eos/kthStiffenedGas.hpp"
#include "eos/nPhase.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "nPhaseAllaireAdvection.hpp"
#include "utilities/constants.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"
#include <petsc/private/dmpleximpl.h>
#include "finiteVolume/faceInterpolant.hpp"
#include "finiteVolume/cellInterpolant.hpp"

#define saveData 1

namespace ablate::finiteVolume::processes {

  // Every time the mesh changes.
  void ablate::finiteVolume::processes::NPhaseIntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {}


  ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharp(const PetscReal gammaFactor, const PetscReal epsilon, const PetscReal p0, const PetscInt preGauss, const PetscInt postGauss) : gammaFactor(gammaFactor), epsilon(epsilon), p0(p0), preGauss(preGauss), postGauss(postGauss) {}

  ablate::finiteVolume::processes::NPhaseIntSharp::~NPhaseIntSharp() {}

  // Run once per simulation
  void ablate::finiteVolume::processes::NPhaseIntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

//        flow.EnableSlopeLimiterFor(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
//        flow.EnableSlopeLimiterFor(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);

    subDomain = flow.GetSubDomainPtr();

    DMPlexGetMinRadius(subDomain->GetDM(), &h) >> utilities::PetscUtilities::checkError;

    // Check for the required fields
    std::vector<std::string> requiredFieldList;
    std::vector<ablate::domain::FieldLocation> requiredLocationList;

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
    requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
    requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE);
    requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::UI);
    requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::EPSILONK);
    requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::PRESSURE);
    requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::RHOK);
    requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::AIJ);
    requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

    std::size_t k = 0;
    for (auto fieldName : requiredFieldList) {
      if (!(subDomain->ContainsField(fieldName))) {
        throw std::runtime_error("ablate::finiteVolume::processes::IntSharp expects a "+ fieldName +" field to be defined.");
      }
      const ablate::domain::Field field = subDomain->GetField(fieldName);
      if (field.location != requiredLocationList[k++]) {
        throw std::runtime_error("ablate::finiteVolume::processes::IntSharp: "+ fieldName +" is in the incorrect location.");
      }
    }

    int advLoc = flow.FindProcessLocation<ablate::finiteVolume::processes::NPhaseAllaireAdvection>();
    int sharpLoc = flow.FindProcessLocation<ablate::finiteVolume::processes::NPhaseIntSharp>();

    if (advLoc < 0 || advLoc > sharpLoc) throw std::runtime_error("The process ablate::finiteVolume::processes::NPhaseAllaireAdvection must be before ablate::finiteVolume::processes::NPhaseIntSharp");

    // Pull the EOS from the advection process
    auto baseEOS = flow.FindProcess<ablate::finiteVolume::processes::NPhaseAllaireAdvection>()->GetEOS();
    eosNPhase.reserve(baseEOS.size());

    for (const auto& individualEOS : baseEOS) {
      auto stiffEOS = std::dynamic_pointer_cast<ablate::eos::KthStiffenedGas>(individualEOS);
      if (stiffEOS) eosNPhase.push_back(stiffEOS);
      else throw std::runtime_error("EOS must be KthStiffnedGas in nPhaseIntSharp.");
    }

#if 1
    // Continuous flux function
    flow.RegisterRHSFunction(NPhaseIntSharpPointFlux, this,
      {ablate::finiteVolume::NPhaseFlowFields::ALPHAK, ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK, ablate::finiteVolume::NPhaseFlowFields::ALLAIRE},
      {ablate::finiteVolume::NPhaseFlowFields::ALPHAK},
      {ablate::finiteVolume::NPhaseFlowFields::UI,        // 0
       ablate::finiteVolume::NPhaseFlowFields::RHOK,      // 1
       ablate::finiteVolume::NPhaseFlowFields::PRESSURE,  // 2
       ablate::finiteVolume::NPhaseFlowFields::AIJ,       // 3
       ablate::finiteVolume::NPhaseFlowFields::EPSILONK   // 4
      });
#else
    PetscPrintf(PETSC_COMM_WORLD, "Turning off flux: %s::%d\n", __FILE__, __LINE__);
#endif
    auto preStepGamma = std::bind(&ablate::finiteVolume::processes::NPhaseIntSharp::UpdateGamma, this, std::placeholders::_1, std::placeholders::_2);
    flow.RegisterPreStep(preStepGamma);


    auto preStep = std::bind(&ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharpPreSharp, this, std::placeholders::_1, std::placeholders::_2);
    flow.RegisterPreStep(preStep);
  }



  void Rpq(std::size_t p, const PetscReal h, const PetscInt dim, std::size_t nPhases, const PetscReal eps, const PetscReal gamma, const PetscReal *faceAlpha, const PetscReal *faceGAlpha, const PetscReal *faceGAij, PetscReal *r) {

    for (PetscInt d = 0; d < dim; ++d) r[d] = eps * faceGAlpha[p*dim + d];

    for (std::size_t q = 0; q < nPhases; ++q) {

      if (p == q) continue;

      // Gradient of A_{ij} at the face
      const PetscReal *gpq = &faceGAij[(p * nPhases + q) * dim];

      // Magnitude of the gradient
      const PetscReal mag = utilities::MathUtilities::MagVector(dim, gpq);

      for (PetscInt d = 0; d < dim; ++d) r[d] -= faceAlpha[p] * faceAlpha[q] * gpq[d] / ( mag + h*h );
    }

    const PetscReal hGamma = PetscTanhReal(faceAlpha[p] * (1.0 - faceAlpha[p]) * 5.5e5) * gamma;
    for (PetscInt d = 0; d < dim; ++d) r[d] *= hGamma;

  }


  /*
    This is based on a combination of
      "A conservative diffuse-interface method for compressible two-phase flows" by Jain, Mani, and Moin [1] and
      "A conservative second order phase field model for simulation of N-phase flows" by Mirjalilia and Mani [2].

    Specifically, it replaces the equation inside the RHS divergence of Eq. (75) in Ref. [1] by the equation
      inside the RHS divergence of Eq. (2) in Ref[2].

    Note that the formulation in the paper is on the RHS of the advection equations. Since the flux assumes
      that it's on the LHS we must use the negative of the terms in the paper.

    It is also assumed that flux is zero when this function is entered.

  */

  PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharpPointFlux(PetscInt dim, const PetscFVFaceGeom* fg,
  const PetscInt uOff[], const PetscInt uOff_x[],
  const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar field[], const PetscScalar grad[],
  const PetscInt aOff[], const PetscInt aOff_x[],
  const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar aux[], const PetscScalar gradAux[],
  PetscScalar flux[], void* ctx) {

    PetscFunctionBegin;

    auto process = (NPhaseIntSharp *)ctx;
    std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> eosNPhase = process->eosNPhase;
    const PetscReal   eps = process->epsilon;
    const PetscReal gamma = process->gamma;//
    const std::size_t nPhases = eosNPhase.size();
    const PetscReal h = process->h;

    // Solution variables
    const PetscReal     *faceAlpha = &field[uOff[0]];      // alpha interpolated to the face
    const PetscReal    *faceGAlpha = &grad[uOff_x[0]];     // gradient of alpha on the face

    // Aux variables
    const PetscReal     *velL = &auxL[aOff[0]];           // velocity at face
    const PetscReal     *velR = &auxR[aOff[0]];           // velocity at face
    const PetscReal    *rhokL = &auxL[aOff[1]];           // phase density of left cell
    const PetscReal    *rhokR = &auxR[aOff[1]];           // phase density of right cell
    const PetscReal        PL =  auxL[aOff[2]];           // interpolated pressure at the face
    const PetscReal        PR =  auxR[aOff[2]];           // interpolated pressure at the face
    const PetscReal *faceGAij = &gradAux[aOff_x[3]];      // gradient of pairwise alpha at the face
    const PetscReal    *eIntL = &auxL[aOff[4]];           // phase internal energy of left cell
    const PetscReal    *eIntR = &auxR[aOff[4]];           // phase internal energy of right cell

    // Kinetic energy
    const PetscReal keL = 0.5 * utilities::MathUtilities::DotVector(dim, velL, velL);
    const PetscReal keR = 0.5 * utilities::MathUtilities::DotVector(dim, velR, velR);

    const PetscReal u_n = utilities::MathUtilities::DotVector(dim, &aux[aOff[0]], fg->normal);

    PetscArrayzero(flux, 2*nPhases + dim + 1) >> utilities::PetscUtilities::checkError;

    for (std::size_t p = 0; p < nPhases; ++p) {

      PetscReal r[3] = {0, 0, 0};

      Rpq(p, h, dim, nPhases, eps, gamma, faceAlpha, faceGAlpha, faceGAij, r);

      // alphaK
      flux[p] = -utilities::MathUtilities::DotVector(dim, r, fg->normal);

      // alphaK * rhoK
      if (flux[p] > 0) flux[p + nPhases] = rhokL[p] * flux[p];
      else             flux[p + nPhases] = rhokR[p] * flux[p];

      // Energy
      if (flux[p] > 0) flux[2 * nPhases + ablate::finiteVolume::NPhaseFlowFields::RHOE] += (rhokL[p] * (eIntL[p] + keL) + PL) * flux[p];
      else             flux[2 * nPhases + ablate::finiteVolume::NPhaseFlowFields::RHOE] += (rhokR[p] * (eIntR[p] + keR) + PR) * flux[p];

      // Momentum
      PetscReal rho0;
      if (u_n > 0) rho0 = rhokL[p];
      else         rho0 = rhokR[p];
      for (PetscInt d = 0; d < dim; ++d) flux[2 * nPhases + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] -= rho0 * r[d] * u_n;


    }

    PetscFunctionReturn(PETSC_SUCCESS);

  }


  PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::AdvectionFlux(
      PetscInt dim, const PetscFVFaceGeom* fg,
      const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[],
      const PetscInt aOff[], const PetscScalar auxL[], const PetscScalar auxR[],
      PetscScalar flux[], void* ctx) {

      PetscFunctionBegin;


      auto process = (NPhaseIntSharp *)ctx;
      std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> eosNPhase = process->eosNPhase;
      const std::size_t nPhases = eosNPhase.size();

      const PetscReal  *alphaL = &fieldL[uOff[0]];     // VOF
      const PetscReal  *alphaR = &fieldR[uOff[0]];     // VOF

      PetscReal u[2] = {1.0, 0.0};
      PetscReal u_n = utilities::MathUtilities::DotVector(dim, u, fg->normal);
      for (std::size_t i = 0; i < nPhases; ++i) {
        // Volume fraction
        if (u_n > 0) flux[i] = alphaL[i] * u_n;
        else         flux[i] = alphaR[i] * u_n;
      }



      PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::SharpeningFlux(
      PetscInt dim, const PetscFVFaceGeom* fg,
      const PetscInt uOff[], const PetscInt uOff_x[],
      const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar field[], const PetscScalar grad[],
      const PetscInt aOff[], const PetscInt aOff_x[],
      const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar aux[], const PetscScalar gradAux[],
      PetscScalar flux[], void* ctx) {

      PetscFunctionBegin;

      auto process = (NPhaseIntSharp *)ctx;
      std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> eosNPhase = process->eosNPhase;
      const PetscReal   eps = process->epsilon;
      const PetscReal gamma = process->gamma;//
      const std::size_t nPhases = eosNPhase.size();
      const PetscReal h = process->h;


      const PetscReal  *faceAlpha = &field[uOff[0]];
      const PetscReal *faceGAlpha = &grad[uOff_x[0]];
      const PetscReal   *faceGAij = &gradAux[aOff_x[0]];


      for (std::size_t p = 0; p < nPhases; ++p) {

        PetscReal r[3] = {0, 0, 0};

        Rpq(p, h, dim, nPhases, eps, gamma, faceAlpha, faceGAlpha, faceGAij, r);

        flux[p] = -utilities::MathUtilities::DotVector(dim, r, fg->normal);

      }


      PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::UpdateGamma(TS flowTS, ablate::solver::Solver &solver) {

    PetscFunctionBegin;

    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);
    const PetscInt dim = subDomain->GetDimensions();
    DM dm = subDomain->GetDM();
    DM auxDM = subDomain->GetAuxDM();
    Vec X;
    Vec auxVec = subDomain->GetAuxVector();
    const PetscScalar *auxArray, *xArray;
    const ablate::domain::Field& alphaField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
    const ablate::domain::Field&   velField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::UI);
    const std::size_t nPhases = alphaField.numberComponents;


    // Maximum velocity associated with an interface
    PetscReal uMax = PETSC_MIN_REAL;

    PetscFunctionBegin;

    PetscCall(TSGetSolution(flowTS, &X));

    PetscCall(VecGetArrayRead(X, &xArray));
    PetscCall(VecGetArrayRead(auxVec, &auxArray));

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      const PetscInt cell = cellRange.GetPoint(c);

      const PetscScalar *alpha;
      PetscCall(DMPlexPointLocalFieldRead(dm, cell, alphaField.id, xArray, &alpha));

      if (!alpha) continue; // Not owned by this rank

      // Determine if this is near an interface
      PetscReal max_ai_aj = PETSC_MIN_REAL;
      for (std::size_t i = 0; i < nPhases; ++i)
        for (std::size_t j = i + 1; j < nPhases; ++j)
          max_ai_aj = PetscMax(max_ai_aj, alpha[i] * alpha[j]);

      if (max_ai_aj < 1e-10) continue; // Not near an interface

      const PetscScalar *vel;
      PetscCall(DMPlexPointLocalFieldRead(auxDM, cell, velField.id, auxArray, &vel));

      uMax = PetscMax(uMax, utilities::MathUtilities::MagVector(dim, vel));
    }

    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &uMax, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));

    gamma = gammaFactor * uMax;

//    PetscPrintf(PETSC_COMM_WORLD, "%s::%d\t%+e\n", __FILE__, __LINE__, gamma);

    PetscFunctionReturn(PETSC_SUCCESS);


  }

  // Sharpen the interface before anything else is done.
  PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharpPreSharp(TS flowTS, ablate::solver::Solver &solver) {

    PetscFunctionBegin;

    // Only run this once
    if (preStageHasRun) PetscFunctionReturn(PETSC_SUCCESS);

    PetscPrintf(PETSC_COMM_WORLD, "Starting initial sharpening.\n");

    // Gamma may have been set to a non-unitary number. Re-set it to zero.
    const PetscReal gamma0 = gamma;
    gamma = 1;


    DM dm = subDomain->GetDM();
    const PetscInt dim = subDomain->GetDimensions();

    ablate::domain::Range faceRange, cellRange;
    solver.GetFaceRange(faceRange);
    solver.GetCellRangeWithoutGhost(cellRange);

    Vec faceGeomVec, cellGeomVec;
    PetscReal h;
    PetscCall(DMPlexGetGeometryFVM(dm, &faceGeomVec, &cellGeomVec, &h));

    // Solution vectors
    Vec X, locX;
    PetscReal *xArray;
    PetscCall(TSGetSolution(flowTS, &X));
    PetscCall(DMGetLocalVector(dm, &locX));
    PetscCall(DMGlobalToLocal(dm, X, INSERT_VALUES, locX));

    // RHS vectors
    Vec F, locF;
    PetscCall(DMGetLocalVector(dm, &locF));
    PetscCall(DMGetGlobalVector(dm, &F));

    // Aux variables
    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();

    const ablate::domain::Field&    alphaField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
    const ablate::domain::Field& rhoAlphaField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
    const ablate::domain::Field&  allaireField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE);
    const ablate::domain::Field&      aijField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::AIJ);

    const std::size_t nPhases = alphaField.numberComponents;

    // Create the function description that will be used with the faceInterpolant
    FaceInterpolant::ContinuousFluxFunctionDescription faceDescription;
    faceDescription.function = SharpeningFlux;
    faceDescription.context = this;
    faceDescription.updateFields.push_back(alphaField.id);
    faceDescription.inputFields.push_back(alphaField.id);
    faceDescription.auxFields.push_back(aijField.id);
    std::vector<FaceInterpolant::ContinuousFluxFunctionDescription> allFaceFunctions;
    allFaceFunctions.push_back(faceDescription);

    // Face interpolant to calculate the RHS
    std::unique_ptr<FaceInterpolant> faceInterpolant = std::make_unique<FaceInterpolant>(subDomain, solver.GetRegion(), faceGeomVec, cellGeomVec);
    faceInterpolant->SetUseGaussianConvolution(PETSC_TRUE);

    // Smooth the fields first
    auto gaussConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(dm, 1, dim, dim);

    for (PetscInt pg = 0; pg < preGauss; ++pg) {
      PetscCall(VecGetArray(X, &xArray));
      for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscReal cell = cellRange.GetPoint(c);
        PetscScalar *vals;
        PetscCall(DMPlexPointGlobalFieldRef(dm, cell, alphaField.id, xArray, &vals));
        if (vals) {
          gaussConv->Evaluate(cell, nullptr, dm, alphaField.id, locX, 0, nPhases, vals);
          PetscReal aSum = 0;
          for (std::size_t k = 0; k < nPhases; ++k) aSum += vals[k];
          for (std::size_t k = 0; k < nPhases; ++k) vals[k] /= aSum;
        }
      }
      PetscCall(VecRestoreArray(X, &xArray));
      PetscCall(DMGlobalToLocal(dm, X, INSERT_VALUES, locX));
    }

#if saveData

    int rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    char fname[255];
    sprintf(fname, "alpha%05d.txt", 0);
    FILE *f1;
    if (rank==0) f1 = fopen(fname, "w");
    else         f1 = fopen(fname, "a");
    VecGetArray(X, &xArray);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      const PetscReal cell = cellRange.GetPoint(c);
      PetscReal x[2];
      const PetscScalar *vals;
      DMPlexPointGlobalFieldRead(dm, cell, alphaField.id, xArray, &vals);

      if (vals) {

        DMPlexPointGeometricData(dm, cell, NULL, x, NULL);
        PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+e\t%+e\t", x[0], x[1]);
        for (std::size_t k = 0; k < nPhases; ++k) PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+.16e\t", vals[k]);

        DMPlexPointGlobalFieldRead(dm, cell, rhoAlphaField.id, xArray, &vals);
        for (std::size_t k = 0; k < nPhases; ++k) PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+.16e\t", vals[k]);

        DMPlexPointGlobalFieldRead(dm, cell, allaireField.id, xArray, &vals);
        PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+.16e\t%+.16e\t%+.16e\n", vals[0], vals[1], vals[2]);

      }
    }
    VecRestoreArray(X, &xArray);
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, f1));
    fclose(f1);

#endif



    PetscReal dk[nPhases], dk0[nPhases];
    for (std::size_t k = 0; k < nPhases; ++k) dk[k] = PETSC_MAX_REAL;
    PetscReal maxDkDiff = -PETSC_MAX_REAL;
    PetscReal minDk = PETSC_MAX_REAL;

    PetscInt iter = 0;
    do {

      ++iter;

      PetscScalar *aArray;
      VecGetArray(auxVec, &aArray);
      VecGetArray(locX, &xArray);
      for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt cell = cellRange.GetPoint(c);

        const PetscScalar *alpha;
        DMPlexPointLocalFieldRead(dm, cell, alphaField.id, xArray, &alpha);

        PetscScalar *aij;
        DMPlexPointLocalFieldRef(auxDM, cell, aijField.id, aArray, &aij);

        for (std::size_t p = 0; p < nPhases; ++p) {
          for (std::size_t q = 0; q < nPhases; ++q) {
            if (p==q) {
              aij[p*nPhases + q] = 0.5;
            }
            else {
              PetscReal denom = alpha[p] + alpha[q] + PETSC_SQRT_MACHINE_EPSILON;
//              PetscReal value = (denom > PETSC_SMALL) ? (alpha[p] / denom) : 0.0;
              PetscReal value = alpha[p] / denom;
              aij[p*nPhases + q] = value;
            }
          }
        }
      }
      VecRestoreArray(locX, &xArray);
      VecRestoreArray(auxVec, &aArray);


      PetscCall(VecZeroEntries(locF));
      faceInterpolant->ComputeRHS(0.0, locX, auxVec, locF, solver.GetRegion(), allFaceFunctions, cellRange, faceRange, cellGeomVec, faceGeomVec);

      VecScale(locF, 10*h*h);
      PetscCall(DMLocalToGlobal(dm, locF, ADD_VALUES, X));

      /*
        Stopping criteria: See Step 3 in "An enhanced interface-sharpening algorithm for accurate simulation of
          underwater explosion in compressible multiphase flow on complex grids" by Jiang et al.
      */
      VecGetArray(X, &xArray);
      PetscArraycpy(dk0, dk, nPhases) >> utilities::PetscUtilities::checkError;
      PetscArrayzero(dk, nPhases) >> utilities::PetscUtilities::checkError;
      for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt cell = cellRange.GetPoint(c);

        PetscScalar *alpha;
        DMPlexPointGlobalFieldRef(dm, cell, alphaField.id, xArray, &alpha);

        if (alpha) {
          for (std::size_t k = 0; k < nPhases; k++) {

            if (alpha[k] < PETSC_SMALL || alpha[k] > 1 - PETSC_SMALL) continue;

            if (0.5 < alpha[k]) dk[k] += PetscSqr(1.0 - alpha[k]);
            else                dk[k] += PetscSqr(alpha[k]);

          }
        }
      }
      VecRestoreArray(X, &xArray);


      maxDkDiff = -PETSC_MAX_REAL;
      minDk = PETSC_MAX_REAL;
      for (std::size_t k = 0; k < nPhases; k++) {
        MPI_Allreduce(MPI_IN_PLACE, &dk[k], 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD) >> utilities::MpiUtilities::checkError;

        if (dk[k] > 0) {
          maxDkDiff = PetscMax(maxDkDiff, PetscAbsReal((dk[k] - dk0[k]) / dk0[k]));
          minDk = PetscMin(minDk, dk[k]);
        }
      }


      // Re-normalize
      VecGetArray(X, &xArray);
      for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt cell = cellRange.GetPoint(c);

        PetscScalar *alpha;
        DMPlexPointGlobalFieldRef(dm, cell, alphaField.id, xArray, &alpha);

        if (alpha) {

          PetscReal alphaSum = 0;
          for (std::size_t k = 0; k < nPhases; k++) {
            alpha[k] = PetscMax(0, PetscMin(1, alpha[k]));
            alphaSum += alpha[k];
          }

          for (std::size_t k = 0; k < nPhases; k++) alpha[k] /= alphaSum;
        }

      }
      VecRestoreArray(X, &xArray);


#if saveData
      if (iter%10==0) {
        PetscReal nrm;
        VecNorm(locF, NORM_INFINITY, &nrm) >> utilities::PetscUtilities::checkError;
        MPI_Allreduce(MPI_IN_PLACE, &nrm, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD) >> utilities::MpiUtilities::checkError;
        PetscPrintf(PETSC_COMM_WORLD, "%05d: %e\t%+e\t%+e\n", iter, nrm / (10*h*h), maxDkDiff, minDk);
      }

      if (iter%1000==0){
        sprintf(fname, "alpha%05d.txt", iter);
        if (rank==0) f1 = fopen(fname, "w");
        else         f1 = fopen(fname, "a");
        VecGetArray(X, &xArray);

        PetscScalar *fArray;
        VecGetArray(locF, &fArray);

        for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
          const PetscReal cell = cellRange.GetPoint(c);

          const PetscScalar *vals;
          DMPlexPointGlobalFieldRead(dm, cell, alphaField.id, xArray, &vals);

          if (vals) {

            PetscReal x[2];
            DMPlexPointGeometricData(dm, cell, NULL, x, NULL);
            PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+e\t%+e\t", x[0], x[1]);

            for (std::size_t k = 0; k < nPhases; ++k) PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+e\t", vals[k]);

            DMPlexPointLocalFieldRead(dm, cell, alphaField.id, fArray, &vals);
            for (std::size_t k = 0; k < nPhases; ++k) PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+e\t", vals[k]);
            PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "\n");

          }
        }
        VecRestoreArray(X, &xArray);
        VecRestoreArray(locF, &fArray);
        PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, f1));
        fclose(f1);
      }
#endif

      PetscCall(DMGlobalToLocal(dm, X, INSERT_VALUES, locX));

      if (iter < 20) maxDkDiff = 1;

//if (iter == 20) {
//  PetscPrintf(PETSC_COMM_WORLD, "Manually exiting %s at %d\n", __FUNCTION__, __LINE__);
//  maxDkDiff = 0;
//}

    } while (iter <= 5000 && maxDkDiff > 2e-4 && minDk > 0);



    MPI_Barrier(PETSC_COMM_WORLD);

    for (PetscInt pg = 0; pg < postGauss; ++pg) {
      PetscCall(VecGetArray(X, &xArray));
      for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscReal cell = cellRange.GetPoint(c);
        PetscScalar *vals;
        PetscCall(DMPlexPointGlobalFieldRef(dm, cell, alphaField.id, xArray, &vals));
        if (vals) {
          gaussConv->Evaluate(cell, nullptr, dm, alphaField.id, locX, 0, nPhases, vals);
          PetscReal aSum = 0;
          for (std::size_t k = 0; k < nPhases; ++k) aSum += vals[k];
          for (std::size_t k = 0; k < nPhases; ++k) vals[k] /= aSum;
        }
      }
      PetscCall(VecRestoreArray(X, &xArray));
      PetscCall(DMGlobalToLocal(dm, X, INSERT_VALUES, locX));
    }


    /*
        Re-construct the conserved fields:
            SOL: alphak * rhok
            SOL: density * (internal energy + kinetic energy)
            SOL: density * velocity
    */
    const PetscScalar *auxArray;
    VecGetArrayRead(auxVec, &auxArray);
    VecGetArray(X, &xArray);
    // Required primative variables
    const ablate::domain::Field& velField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::UI);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      const PetscReal cell = cellRange.GetPoint(c);

      PetscScalar *alpha;
      DMPlexPointGlobalFieldRef(dm, cell, alphaField.id, xArray, &alpha);

      if (alpha) {

        // Update alphaK
        PetscReal aSum = 0;
        for (std::size_t k = 0; k < nPhases; ++k) {
          alpha[k] = PetscMin(PetscMax(alpha[k], 0), 1);
          aSum += alpha[k];
        }
        for (std::size_t k = 0; k < nPhases; ++k) alpha[k] /= aSum;

        aSum = 0;
        for (std::size_t k = 0; k < nPhases; ++k) aSum += alpha[k];
        if (PetscAbsReal(aSum - 1) > PETSC_SQRT_MACHINE_EPSILON) {
          printf("%e\n", PetscAbsReal(aSum - 1));
          printf("%s::%d\n", __FILE__, __LINE__);
          exit(0);
        }


        PetscScalar *alphaRho;
        DMPlexPointGlobalFieldRef(dm, cell, rhoAlphaField.id, xArray, &alphaRho);


        PetscReal a = 0, b = 0;
        PetscReal mixRho = 0;
        for (std::size_t k = 0; k < nPhases; ++k) {
          const PetscReal rho = eosNPhase[k]->GetReferenceDensity();
          const PetscReal gamma = eosNPhase[k]->GetSpecificHeatRatio();
          const PetscReal P0 = eosNPhase[k]->GetReferencePressure();

          alphaRho[k] = alpha[k] * rho;
          mixRho += alphaRho[k];

          a += alpha[k] / (gamma - 1);
          b += alpha[k] * gamma * P0 / (gamma - 1);
        }

        // Mixture internal energy
        PetscReal mixEnergy = (p0 * a + b);

        // Kinetic energy
        const PetscScalar *vel;
        DMPlexPointLocalFieldRead(auxDM, cell, velField.id, auxArray, &vel);
        PetscReal ke = 0;
        for (PetscInt d = 0; d < dim; ++d) ke += vel[d] * vel[d];
        ke *= 0.5;

        PetscScalar *allaire;
        DMPlexPointGlobalFieldRef(dm, cell, allaireField.id, xArray, &allaire);
        allaire[ablate::finiteVolume::NPhaseFlowFields::RHOE] = mixEnergy + mixRho * ke;
        for (PetscInt d = 0; d < dim; ++d) allaire[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] = mixRho * vel[d];
      }
    }
    VecRestoreArray(X, &xArray);

    PetscCall(DMGlobalToLocal(dm, X, INSERT_VALUES, locX));

    // Make sure the AUX variable are also updated
    auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
    fvSolver.UpdateAuxFields(NAN, locX, auxVec);

#if saveData

    sprintf(fname, "alpha_final.txt");
    if (rank==0) f1 = fopen(fname, "w");
    else         f1 = fopen(fname, "a");
    VecGetArray(X, &xArray);
    VecGetArrayRead(auxVec, &auxArray);
const ablate::domain::Field& pField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::PRESSURE);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      const PetscReal cell = cellRange.GetPoint(c);

      const PetscScalar *vals;
      DMPlexPointGlobalFieldRead(dm, cell, alphaField.id, xArray, &vals);

      if (vals) {

        /*
          x, y: 1,2
          alpha: 3, 4
          rhoAlpha: 5,6
          rhoE: 7
          rhoU: 8, 9
          p: 10
        */

        PetscReal x[2];
        DMPlexPointGeometricData(dm, cell, NULL, x, NULL);
        PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+e\t%+e\t", x[0], x[1]);


        for (std::size_t k = 0; k < nPhases; ++k) PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+.16e\t", vals[k]);

        DMPlexPointGlobalFieldRead(dm, cell, rhoAlphaField.id, xArray, &vals);
        for (std::size_t k = 0; k < nPhases; ++k) PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+.16e\t", vals[k]);

        DMPlexPointGlobalFieldRead(dm, cell, allaireField.id, xArray, &vals);
        PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+.16e\t%+.16e\t%+.16e\t", vals[0], vals[1], vals[2]);

        DMPlexPointGlobalFieldRead(auxDM, cell, pField.id, auxArray, &vals);
        PetscSynchronizedFPrintf(PETSC_COMM_WORLD, f1, "%+.16e\n", vals[0]);
      }
    }
    VecRestoreArray(X, &xArray);
    VecRestoreArrayRead(auxVec, &auxArray);
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, f1));
    MPI_Barrier(PETSC_COMM_WORLD);
    fclose(f1);

#endif

    solver.RestoreRange(faceRange);
    solver.RestoreRange(cellRange);

    PetscCall(DMRestoreLocalVector(dm, &locX));
    PetscCall(DMRestoreLocalVector(dm, &locF));
    PetscCall(DMRestoreGlobalVector(dm, &F));

    preStageHasRun = PETSC_TRUE;

    // Old value
    gamma = gamma0;

    PetscPrintf(PETSC_COMM_WORLD, "Finished initial sharpening.\n");

    PetscFunctionReturn(0);
}

} // ablate::finiteVolume::processes


#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process,
    ablate::finiteVolume::processes::NPhaseIntSharp,
    "N-phase interface regularization term",
    ARG(PetscReal, "gammaFactor", "gammaFactor, gamma = gammaFactor * |vel|"),
    ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)"),
    ARG(PetscReal, "p0", "pressure, initial pressure to use when reconstructing conserved fields after pre-stage sharpening"),
    OPT(PetscInt, "preGauss", "preGauss, apply this number of Gaussian kernel smoothing operations before initial sharpening. Default is 0"),
    OPT(PetscInt, "postGauss", "postGauss, apply this number of Gaussian kernel smoothing operations after initial sharpening. Default is 0")
    );
