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

  // Every time the mesh changes.
  void ablate::finiteVolume::processes::NPhaseIntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {}


  ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharp(PetscReal Gamma, PetscReal epsilon) : Gamma(Gamma), epsilon(epsilon) {}

  ablate::finiteVolume::processes::NPhaseIntSharp::~NPhaseIntSharp() {}

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

    // Continuous flux function
    flow.RegisterRHSFunction(NPhaseIntSharpPointFlux, this,
      {ablate::finiteVolume::NPhaseFlowFields::ALPHAK, ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK, ablate::finiteVolume::NPhaseFlowFields::ALLAIRE},
      {ablate::finiteVolume::NPhaseFlowFields::ALPHAK},
      {ablate::finiteVolume::NPhaseFlowFields::UI, ablate::finiteVolume::NPhaseFlowFields::RHOK, ablate::finiteVolume::NPhaseFlowFields::PRESSURE, ablate::finiteVolume::NPhaseFlowFields::EPSILONK});

  }

  /*
    This is based on "A conservative diffuse-interface method for compressible two-phase flows" by Jain, Mani, and Moin

    Note that the formulation in the paper is on the RHS of the advection equations. Since the flux assumes
      that it's on the LHS we must use the negative of the terms in the paper.

    It is also assumed that flux is zero when this function is entered.

  */
//  static PetscInt cnt = 0;
  PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharpPointFlux(
      PetscInt dim, const PetscFVFaceGeom* fg,
      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
      PetscScalar flux[], void* ctx) {

    PetscFunctionBegin;
//++cnt;
    auto process = (NPhaseIntSharp *)ctx;
    std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> eosNPhase = process->eosNPhase;
    const PetscReal Gamma = process->Gamma;
    const PetscReal epsilon = process->epsilon;
    const std::size_t nPhases = eosNPhase.size();

    // Extracted variables in the order they are passed in dring the Register call
    const PetscReal  *alpha = &field[uOff[0]]; // VOF
    const PetscReal    *vel = &aux[aOff[0]];   // Velocity
    const PetscReal    *rho = &aux[aOff[1]];   // Phase density
    const PetscReal       P =  aux[aOff[2]];   // Pressure
    const PetscReal   *eInt = &aux[aOff[3]];   // Phase internal energy

    // Kinetic energy
    const PetscReal ke = 0.5 * utilities::MathUtilities::DotVector(dim, vel, vel);

    // Velocity dot area normal
    const PetscReal u_n = utilities::MathUtilities::DotVector(dim, vel, fg->normal);

    PetscReal aTotal = 0;

    for (std::size_t k = 0; k < nPhases; ++k) {
      // Volume fraction
      if (alpha[k] < PETSC_MACHINE_EPSILON || alpha[k] > 1 - PETSC_MACHINE_EPSILON) continue;

      const PetscReal *alphaGrad = &grad[uOff_x[0] + k*dim]; // Gradient of the current VOF field

      PetscReal mag = utilities::MathUtilities::MagVector(dim, alphaGrad);

      // grad of VOF dot area normal
      const PetscReal gradVOF_n = utilities::MathUtilities::DotVector(dim, alphaGrad, fg->normal);

      // RHS of Eq. (75)
      const PetscReal a = Gamma * (epsilon - alpha[k] * (1 - alpha[k]) / mag);
      aTotal += a * gradVOF_n;

      // alpha_k
      flux[k] = -a * gradVOF_n;

      // rho_k * alpha_k
//      const PetscReal rho0 = eosNPhase[k]->GetReferenceDensity();
      const PetscReal rho0 = rho[k];
      flux[nPhases + k] = -rho0 * flux[k];

      // Energy: Note that flux[k] and flux[nPhases + k] already have the negative sign, so it's not needed here
      const PetscReal rhoH = rho[k] * eInt[k] + P;
      flux[2*nPhases + ablate::finiteVolume::NPhaseFlowFields::RHOE] += flux[nPhases + k] * ke;
      flux[2*nPhases + ablate::finiteVolume::NPhaseFlowFields::RHOE] += rhoH * flux[k];

//      const PetscReal cRatio = eosNPhase[k]->GetSpecificHeatRatio();
//      const PetscReal P0     = eosNPhase[k]->GetReferencePressure();
//      const PetscReal rhoH   = (aux[aOff[2]] + P0)*cRatio/(cRatio-1);


      // Momentum
      for (PetscInt d = 0; d < dim; ++d) flux[2*nPhases + ablate::finiteVolume::NPhaseFlowFields::RHOU + d] -= rho0 * a * alphaGrad[d] * u_n;
    }

    if (PetscAbsReal(aTotal) > PETSC_SMALL) {
//      printf("%d\n", cnt);
      for (std::size_t k = 0; k < nPhases; ++k) printf("%+e\t", field[uOff[0] + k]);
      printf("\n");
//      printf("%+e\n", Gammak[0] - Gammak[1]);
//      printf("%+e\n", epsilonk[0] - epsilonk[1]);
//      printf("%+e\n", field[uOff[0] + 0] + field[uOff[0] + 1] - 1);
//      printf("%+e\t%+e\t%+e\n", field[uOff[0]], field[uOff[0] + 1], field[uOff[0]] + field[uOff[0] + 1] - 1);
//      printf("%+e\t%+e\n", grad[uOff_x[0] + 0*dim + 0], grad[uOff_x[0] + 0*dim + 1]);
//      printf("%+e\t%+e\n", grad[uOff_x[0] + 1*dim + 0], grad[uOff_x[0] + 1*dim + 1]);
//      printf("%+e\t%+e\n", utilities::MathUtilities::MagVector(dim, &grad[uOff_x[0] + 0*dim]),
//                           utilities::MathUtilities::MagVector(dim, &grad[uOff_x[0] + 1*dim]));
      printf("%+e\n", aTotal);
      throw std::runtime_error("Values of a do no sum to zero.");
    }


//fprintf(f1, "%+e\t%+e\t", fg->centroid[0], fg->centroid[1]);
//loc = 0;
//for (std::size_t k = 0; k < nPhases; ++k) fprintf(f1, "%+e\t", flux[loc++]); // alpha
//for (std::size_t k = 0; k < nPhases; ++k) fprintf(f1, "%+e\t", flux[loc++]); // alpha*rho
//fprintf(f1, "%+e\t", flux[loc++]); // Energy
//for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%+e\t", flux[loc++]); // momentum
//fprintf(f1, "\n");
//fprintf(f1, "%+e\t%+e\t", field[uOff[0]], field[uOff[0]+1]);
//for (std::size_t k = 0; k < nPhases;++k) {
//  for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%+e\t", grad[uOff_x[0] + k*dim + d]);
//}
//fprintf(f1,"\n");




    PetscFunctionReturn(PETSC_SUCCESS);

  }


} // ablate::finiteVolume::processes


#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process,
    ablate::finiteVolume::processes::NPhaseIntSharp,
    "N-phase interface regularization term",
    ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
    ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)")
    );
