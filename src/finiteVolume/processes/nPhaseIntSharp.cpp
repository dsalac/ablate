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
#include "utilities/petscUtilities.hpp"
#include <petsc/private/dmpleximpl.h>

namespace ablate::finiteVolume::processes {

    // Every time the mesh changes
    void ablate::finiteVolume::processes::NPhaseIntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {


      // Create any required structures. This must be done AFTER Setup() as the vectors haven't been created when Setup() is called
      if (isPostStep) {

        // Create a DM just for the alphaK data
        auto field = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
        auto entireDm = subDomain->GetFieldDM(field);
        auto entireVec = subDomain->GetSolutionVector();

        if (subIS) ISDestroy(&subIS) >> utilities::PetscUtilities::checkError;
        if (subDM) DMDestroy(&subDM) >> utilities::PetscUtilities::checkError;

        DMCreateSubDM(entireDm, 1, &field.id, &subIS, &subDM) >> utilities::PetscUtilities::checkError;

        if (subGlobVec) VecDestroy(&subGlobVec) >> utilities::PetscUtilities::checkError;
        DMCreateGlobalVector(subDM, &subGlobVec) >> utilities::PetscUtilities::checkError;

        if (subLocVec) VecDestroy(&subLocVec) >> utilities::PetscUtilities::checkError;
        DMCreateLocalVector(subDM, &subLocVec) >> utilities::PetscUtilities::checkError;

        // Now create the scatter to pull the data. Relevent portions copied from VecGetSubVectorThroughVecScatter_Private
        if (subScatter) VecScatterDestroy(&subScatter) >> utilities::PetscUtilities::checkError;
        VecScatterCreate(entireVec, subIS, subGlobVec, NULL, &subScatter) >> utilities::PetscUtilities::checkError;
      }


    }


    ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharp(const std::vector<PetscReal>& Gammak, const std::vector<PetscReal>& epsilonk, const std::vector<PetscInt>& flipPhiTildek, const bool isPostStep) : Gammak(Gammak), epsilonk(epsilonk), flipPhiTildek(flipPhiTildek), isPostStep(isPostStep){}

    ablate::finiteVolume::processes::NPhaseIntSharp::~NPhaseIntSharp() {
      if (subIS) ISDestroy(&subIS) >> utilities::PetscUtilities::checkError;
      if (subDM) DMDestroy(&subDM) >> utilities::PetscUtilities::checkError;
      if (subGlobVec) VecDestroy(&subGlobVec) >> utilities::PetscUtilities::checkError;
      if (subLocVec) VecDestroy(&subLocVec) >> utilities::PetscUtilities::checkError;
      if (subScatter) VecScatterDestroy(&subScatter) >> utilities::PetscUtilities::checkError;

    }


    // Run once per simulation
    void ablate::finiteVolume::processes::NPhaseIntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

//        flow.EnableSlopeLimiterFor(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
//        flow.EnableSlopeLimiterFor(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);

        subDomain = flow.GetSubDomainPtr();

        // Check for the required fields
        std::vector<std::string> requiredFieldList;
        std::vector<ablate::domain::FieldLocation> requiredLocationList;

        requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
        requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

        if (isPostStep) {
          requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
          requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

          requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE);
          requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

          requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::UI);
          requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

          requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::RHO);
          requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

          requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::RHOK);
          requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

          requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::EPSILONK);
          requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);
        }
        else {
          requiredFieldList.push_back(FSHARPK_FIELD);
          requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);
        }


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


        // Now register all of the pre/post stage functions
        if (isPostStep) {
          auto postStep = [this](TS ts, ablate::solver::Solver &solver) { this->NPhaseIntSharpPostStep(ts, solver, this); };
          flow.RegisterPreStep(postStep);

        }
        else {
          auto preStage = [this](TS ts, ablate::solver::Solver &solver, PetscReal stagetime) { this->NPhaseIntSharpPreStage(ts, solver, stagetime, this); };
          flow.RegisterPreStage(preStage);

          flow.RegisterRHSFunction(NPhaseIntSharpPointSource, (void*)&(subDomain->GetField(FSHARPK_FIELD).numberComponents),
                {ablate::finiteVolume::NPhaseFlowFields::ALPHAK}, {ablate::finiteVolume::NPhaseFlowFields::ALPHAK}, {FSHARPK_FIELD});
        }
    }

    PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharpPreStage(TS ts, ablate::solver::Solver &solver, PetscReal stagetime,
        ablate::finiteVolume::processes::NPhaseIntSharp* process) {
      PetscFunctionBegin;

      std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
      subDomain->UpdateSolutionLocalVector();

      DM                     dm = subDomain->GetDM();
      Vec                    solLocalVec = subDomain->GetSolutionLocalVector();
      const PetscInt         fsharpID = subDomain->GetField(FSHARPK_FIELD).id;
      const PetscInt         nPhases = subDomain->GetField(FSHARPK_FIELD).numberComponents;
      const PetscInt         dim = subDomain->GetDimensions();
      auto                   fsharpAccessor = subDomain->GetAuxAccessor(FSHARPK_FIELD);
      auto                   alphaAccessor = subDomain->GetConstSolutionAccessor(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
      std::vector<PetscReal> Gammak       = process->Gammak;
      std::vector<PetscReal> epsilonk     = process->epsilonk;
      std::vector<PetscInt>  flipPhiTildek = process->flipPhiTildek;

      ablate::domain::Range  cellRange;
      solver.GetCellRangeWithoutGhost(cellRange);

      for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt     cell = cellRange.GetPoint(c);
        const PetscScalar *alpha = alphaAccessor[cell];
        PetscScalar       *fsharp = fsharpAccessor[cell];

        for (PetscInt k = 0; k < nPhases; ++k) {
          fsharp[k] = 0;
          if (alpha[k] < PETSC_SMALL || alpha[k] > 1 - PETSC_SMALL) continue;


          const PetscReal a = flipPhiTildek[k] ? 1 - alpha[k] : alpha[k];

          PetscReal g[3];
          DMPlexCellGradFromCell(dm, cell, solLocalVec, fsharpID, k, g) >> utilities::PetscUtilities::checkError;

          const PetscReal nrm = utilities::MathUtilities::MagVector(dim, g);
          fsharp[k] = Gammak[k]*(-a*(1 - a)*(1 - 2*a) + epsilonk[k]*(1 - 2*a)*nrm);
        }


      }

      solver.RestoreRange(cellRange);
//printf("%s::%d\n", __FILE__, __LINE__);
//exit(0);

      PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharpPointSource(PetscInt dim, const PetscReal time, const PetscFVCellGeom *cg,
              const PetscInt *uOff, const PetscScalar *u,
              const PetscInt *aOff, const PetscScalar *a,
              PetscScalar *flux, void *ctx){

        PetscFunctionBegin;

        PetscInt nPhases = *(PetscInt*)ctx;
        for (PetscInt k = 0; k < nPhases; ++k) flux[k] = a[aOff[0] + k];

        PetscFunctionReturn(PETSC_SUCCESS);

    }



    /*
      This basically follows "An enhanced interface-sharpening algorithm for accurate simulation of underwater explosion in compressible
        multiphase flow on complex grids" by Jiang, Tao, Chen, and Dai (2025).

        Step 1: Advect the conserved variables as-is
        Step 2: Obtain the primative variables: rho, rhoK, internal energy, velocity
        Step 3: Sharpen the interface
        Step 4: Re-construct the conserved variables

    */
    PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharpPostStep(TS flowTs, ablate::solver::Solver &solver, ablate::finiteVolume::processes::NPhaseIntSharp* process) {

        PetscFunctionBegin;

        std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
        const PetscInt               nPhases = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK).numberComponents;
        const PetscInt                   dim = subDomain->GetDimensions();
        DM                             subDM = process->subDM;
        Vec                       subGlobVec = process->subGlobVec, subLocVec = process->subLocVec;
        Vec                           solVec = subDomain->GetSolutionVector();
        VecScatter                subScatter = process->subScatter;
        std::vector<PetscReal>  Gammak       = process->Gammak;
        std::vector<PetscReal>  epsilonk     = process->epsilonk;
        std::vector<PetscInt>  flipPhiTildek = process->flipPhiTildek;

        PetscReal h;
        DMPlexGetMinRadius(subDM, &h) >> utilities::PetscUtilities::checkError; // This is 1/2 of the smallest cell size

        ablate::domain::Range  cellRange;
        solver.GetCellRangeWithoutGhost(cellRange);

        // Pull the data and update overlap cells
        VecScatterBegin(subScatter, solVec, subGlobVec, INSERT_VALUES, SCATTER_FORWARD) >> utilities::PetscUtilities::checkError;
        VecScatterEnd(subScatter, solVec, subGlobVec, INSERT_VALUES, SCATTER_FORWARD) >> utilities::PetscUtilities::checkError;

//{
//  FILE *f1 = fopen("Before.txt", "w");
//  PetscScalar *array;
//  VecGetArray(subGlobVec, &array) >> utilities::PetscUtilities::checkError;
//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    const PetscInt cell = cellRange.GetPoint(c);
//    PetscReal x[2];
//    DMPlexPointGeometricData(subDM, cell, NULL, x, NULL);
//    PetscScalar *alpha;
//    DMPlexPointGlobalRef(subDM, cell, array, &alpha) >> utilities::PetscUtilities::checkError;
//    fprintf(f1, "%+e\t%+e\t%+e\t%+e\n", x[0], x[1], alpha[0], alpha[1]);
//  }
//  fclose(f1);
//  VecRestoreArray(subGlobVec, &array) >> utilities::PetscUtilities::checkError;
//}

        PetscInt iter = 0;
        PetscReal maxDiff = PETSC_MAX_REAL;

        PetscReal dk[nPhases], dk0[nPhases];
        for (PetscInt k = 0; k < nPhases; ++k) dk[k] = PETSC_MAX_REAL;

        while (iter < 1000 && maxDiff > 1e-3) {

          ++iter;

          // Update the sub-local vector
          DMGlobalToLocal(subDM, subGlobVec, INSERT_VALUES, subLocVec) >> utilities::PetscUtilities::checkError;

          PetscScalar *array;
          VecGetArray(subGlobVec, &array) >> utilities::PetscUtilities::checkError;


          for (PetscInt k = 0; k < nPhases; ++k) {
            dk0[k] = dk[k];
            dk[k] = 0.0;
          }

          for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
            const PetscInt cell = cellRange.GetPoint(c);

            PetscScalar *alpha;
            DMPlexPointGlobalRef(subDM, cell, array, &alpha) >> utilities::PetscUtilities::checkError;

            if (!alpha) continue; // Not owned by this rank

            for (PetscInt k = 0; k < nPhases; ++k) {

              if (alpha[k] < PETSC_SMALL || alpha[k] > (1 - PETSC_SMALL)) continue;

              const PetscReal a = (flipPhiTildek[k] ? 1 - alpha[k] : alpha[k]);
              PetscReal g[3];
              DMPlexCellGradFromCell(subDM, cell, subLocVec, -1, k, g) >> utilities::PetscUtilities::checkError;

              const PetscReal nrm = utilities::MathUtilities::MagVector(dim, g);

              alpha[k] += h*Gammak[k]*(-a*(1 - a)*(1 - 2*a) + epsilonk[k]*(1 - 2*a)*nrm);

              alpha[k] = PetscMax(0.0, PetscMin(1.0, alpha[k])); // Is this necessary?

              PetscReal p = (0.5 < alpha[k] && alpha[k] < 1);
              PetscReal n = (0.0 < alpha[k] && alpha[k] < 0.5);
              PetscReal ak = (alpha[k] > 0 && alpha[k] < 1 ? alpha[k] : 0);
              dk[k] += p*PetscSqr(1 - ak) + n*PetscSqr(ak);
            }
          }

          VecRestoreArray(subGlobVec, &array) >> utilities::PetscUtilities::checkError;


          maxDiff = -PETSC_MAX_REAL;
          for (PetscInt k = 0; k < nPhases; ++k) {
            if ((dk0[k] - dk[k])/dk0[k] < 0) {
              maxDiff = -1;
              break;
            }
            maxDiff = PetscMax(maxDiff, (dk0[k] - dk[k])/dk0[k]);
          }

        }
//printf("%s::%d, %d\n", __FILE__, __LINE__, iter);

        // Re-scale to account for any overflows and adjust the conserved variables
        PetscScalar *subArray, *solArray;
        VecGetArray(subGlobVec, &subArray) >> utilities::PetscUtilities::checkError;
        VecGetArray(solVec, &solArray) >> utilities::PetscUtilities::checkError;

        DM solDM = subDomain->GetDM();
        ablate::domain::Field alphaField    = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
        ablate::domain::Field alphaRhoField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
        ablate::domain::Field allaireField  = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE);

        for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
          const PetscInt cell = cellRange.GetPoint(c);

          // Adjust the vofs
          PetscScalar *alpha;
          DMPlexPointGlobalRef(subDM, cell, subArray, &alpha) >> utilities::PetscUtilities::checkError;
          if (!alpha) continue; // Not owned by this rank


          PetscReal aSum = 0;
          for (PetscInt k = 0; k < nPhases; ++k) aSum += alpha[k];
          for (PetscInt k = 0; k < nPhases; ++k) alpha[k] /= aSum;

          // Adjust the solution fields
          PetscScalar *alpha0, *alphaRho;
          DMPlexPointGlobalFieldRef(solDM, cell, alphaField.id, solArray, &alpha0) >> utilities::PetscUtilities::checkError;
          DMPlexPointGlobalFieldRef(solDM, cell, alphaRhoField.id, solArray, &alphaRho) >> utilities::PetscUtilities::checkError;

          PetscReal rho0 = 0, rho = 0;
          for (PetscInt k = 0; k < nPhases; ++k) {
            rho0 += alphaRho[k];
            if (alpha0[k] > PETSC_SMALL) alphaRho[k] *= (alpha[k]/alpha0[k]);
            alpha0[k] = alpha[k];
            rho += alphaRho[k];
          }

          PetscScalar *allaire;
          DMPlexPointGlobalFieldRef(solDM, cell, allaireField.id, solArray, &allaire) >> utilities::PetscUtilities::checkError;

          for (PetscInt d = 0; d < dim+1; ++d) allaire[d] *= (rho/rho0);

        }

//{
//  FILE *f1 = fopen("After.txt", "w");
//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    const PetscInt cell = cellRange.GetPoint(c);
//    PetscReal x[2];
//    DMPlexPointGeometricData(subDM, cell, NULL, x, NULL);
//    PetscScalar *vals;
//    DMPlexPointGlobalFieldRef(solDM, cell, alphaField.id, solArray, &vals);
//    fprintf(f1, "%+e\t%+e\t%+e\t%+e\t", x[0], x[1], vals[0], vals[1]);

//    DMPlexPointGlobalFieldRef(solDM, cell, alphaRhoField.id, solArray, &vals);
//    fprintf(f1, "%+e\t%+e\t", vals[0], vals[1]);

//    DMPlexPointGlobalFieldRef(solDM, cell, allaireField.id, solArray, &vals);
//    fprintf(f1, "%+e\t%+e\t%+e\n", vals[0], vals[1], vals[2]);
//  }
//  fclose(f1);
//  printf("%s::%d\n", __FILE__, __LINE__);exit(0);
//}


        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] PreStage completed successfully\n");
        PetscFunctionReturn(0);
    }

}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process,
    ablate::finiteVolume::processes::NPhaseIntSharp,
    "N-phase interface regularization term",
    ARG(std::vector<PetscReal>, "Gammak", "Gamma, velocity scale parameter (approx. umax)"),
    ARG(std::vector<PetscReal>, "epsilonk", "epsilon, interface thickness scale parameter (approx. h)"),
    ARG(std::vector<PetscInt>, "flipPhiTildek", "if 1: phiTilde-->1-phiTilde, if 0: keep phiTilde (set to 1 if primary phase is phi=0 or 0 if phi=1)"),
    OPT(bool, "isPostStep", "False: Apply as a source on the RHS. True: Apply as a post-step operation. Default = False")
    );
