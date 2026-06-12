#include "pressureCorrection.hpp"
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
  void ablate::finiteVolume::processes::PressureCorrection::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {}


  ablate::finiteVolume::processes::PressureCorrection::PressureCorrection(const PetscReal minPressure) : minPressure(minPressure) {}

  ablate::finiteVolume::processes::PressureCorrection::~PressureCorrection() {}

  // Run once per simulation
  void ablate::finiteVolume::processes::PressureCorrection::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

//        flow.EnableSlopeLimiterFor(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
//        flow.EnableSlopeLimiterFor(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);

    subDomain = flow.GetSubDomainPtr();

    // Check for the required fields
    std::vector<std::string> requiredFieldList;
    std::vector<ablate::domain::FieldLocation> requiredLocationList;

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
    requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
    requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE);
    requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::UI);
    requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::PRESSURE);
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

    // Pull the EOS from the advection process
    auto baseEOS = flow.FindProcess<ablate::finiteVolume::processes::NPhaseAllaireAdvection>()->GetEOS();
    eosNPhase.reserve(baseEOS.size());

    for (const auto& individualEOS : baseEOS) {
      auto stiffEOS = std::dynamic_pointer_cast<ablate::eos::KthStiffenedGas>(individualEOS);
      if (stiffEOS) eosNPhase.push_back(stiffEOS);
      else throw std::runtime_error("EOS must be KthStiffnedGas in nPhaseIntSharp.");
    }

    auto PostEval = std::bind(PressureCorrectionPostEvaluate, std::placeholders::_1, std::placeholders::_2, this);
    flow.RegisterPostEvaluate(PostEval);

  }



  void ablate::finiteVolume::processes::PressureCorrection::PressureCorrectionPostEvaluate(TS flowTs, ablate::solver::Solver &solver, void *ctx) {

      auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
      auto process = (PressureCorrection *)ctx;
      ablate::domain::Range cellRange;
      fvSolver.GetCellRangeWithoutGhost(cellRange);
      std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
      const PetscInt dim = subDomain->GetDimensions();
      DM dm = subDomain->GetDM();
      PetscScalar *xArray;
      const ablate::domain::Field&    alphaField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
      const ablate::domain::Field& alphaRhoField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
      const ablate::domain::Field&  allaireField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE);
      const std::size_t                  nPhases = alphaField.numberComponents;
      const PetscReal minPressure = process->minPressure;
      std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> eosNPhase = process->eosNPhase;


      VecGetArray(subDomain->GetSolutionVector(), &xArray) >> utilities::PetscUtilities::checkError;

      PetscReal *gamma, *pi;
      DMGetWorkArray(dm, nPhases, MPIU_REAL, &gamma) >> utilities::PetscUtilities::checkError;
      DMGetWorkArray(dm, nPhases, MPIU_REAL, &pi)  >> utilities::PetscUtilities::checkError;


      for (std::size_t k = 0; k < nPhases; k++) {
        gamma[k] = eosNPhase[k]->GetSpecificHeatRatio();
        pi[k]    = eosNPhase[k]->GetReferencePressure();
      }

      for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt cell = cellRange.GetPoint(c);

        const PetscScalar *alpha;
        DMPlexPointGlobalFieldRead(dm, cell, alphaField.id, xArray, &alpha) >> utilities::PetscUtilities::checkError;
        if (!alpha) continue; // Not owned by this rank

        const PetscScalar *alphaRho;
        DMPlexPointGlobalFieldRead(dm, cell, alphaRhoField.id, xArray, &alphaRho) >> utilities::PetscUtilities::checkError;
        const PetscScalar mixRho = utilities::MathUtilities::SumVector(nPhases, alphaRho);

        PetscScalar *allaire;
        DMPlexPointGlobalFieldRef(dm, cell, allaireField.id, xArray, &allaire);

        PetscReal rhoKE = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
          rhoKE += PetscSqr(allaire[ablate::finiteVolume::NPhaseFlowFields::RHOU + d]);
        }
        rhoKE *= 0.5/mixRho;
        PetscReal rhoIntE = allaire[ablate::finiteVolume::NPhaseFlowFields::RHOE] - rhoKE;

        PetscReal a = 0.0, b = 0;
        for (std::size_t k = 0; k < nPhases; k++) {
          a += alpha[k] / (gamma[k] - 1);
          b += alpha[k] * gamma[k] * pi[k] / (gamma[k] - 1);

        }

        // The decoded pressure
        const PetscReal p = ((rhoIntE - b) / a);

        if (nPhases == 3 && alpha[2] > 0.25) {
          for (PetscInt d = 0; d < dim; ++d) allaire[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] = 0;
        }

        if (p >= minPressure) continue; // Don't need any correction

        // Mixture internal energy based on the minimum pressure
        rhoIntE = (minPressure * a + b);

        // Adjust the energy
        allaire[ablate::finiteVolume::NPhaseFlowFields::RHOE] = rhoIntE + rhoKE;
      }

      DMRestoreWorkArray(dm, nPhases, MPIU_REAL, &gamma)  >> utilities::PetscUtilities::checkError;
      DMRestoreWorkArray(dm, nPhases, MPIU_REAL, &pi)  >> utilities::PetscUtilities::checkError;

      VecRestoreArray(subDomain->GetSolutionVector(), &xArray) >> utilities::PetscUtilities::checkError;


//      subDomain->UpdateSolutionLocalVector();
//      Vec locX = subDomain->GetSolutionLocalVector();
//      PetscReal time;
//      TSGetTime(flowTs, &time);

//      fvSolver.UpdateAuxFields(time, locX, subDomain->GetAuxVector());





//printf("%s::%d\n", __FILE__, __LINE__);
//exit(0);
  }


} // ablate::finiteVolume::processes


#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process,
    ablate::finiteVolume::processes::PressureCorrection,
    "Pressure correction: Do not allow the pressure to go below a given value",
    ARG(PetscReal, "minPressure", "Minimum allowed pressure")
    );
