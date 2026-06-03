#include "nPhaseSurfaceTension.hpp"
#include "eos/kthStiffenedGas.hpp"
#include "eos/nPhase.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "nPhaseAllaireAdvection.hpp"
#include "utilities/petscUtilities.hpp"
#include "utilities/mathUtilities.hpp"
#include <petsc/private/dmpleximpl.h>
#include <vector>

namespace ablate::finiteVolume::processes {

ablate::finiteVolume::processes::NPhaseSurfaceTension::NPhaseSurfaceTension(const std::vector<PetscReal> &sigmaij) : sigmaij(sigmaij) {}

void ablate::finiteVolume::processes::NPhaseSurfaceTension::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    NPhaseSurfaceTension::subDomain = solver.GetSubDomainPtr();
}

void NPhaseSurfaceTension::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

    subDomain = flow.GetSubDomainPtr();

    DMPlexGetMinRadius(subDomain->GetDM(), &h) >> utilities::PetscUtilities::checkError;

    // Check for the required fields
    std::vector<std::string> requiredFieldList;
    std::vector<ablate::domain::FieldLocation> requiredLocationList;

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
    requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE);
    requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

    requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::UI);
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
    int tensionLoc = flow.FindProcessLocation<ablate::finiteVolume::processes::NPhaseSurfaceTension>();

    if (advLoc < 0 || advLoc > tensionLoc) throw std::runtime_error("The process ablate::finiteVolume::processes::NPhaseAllaireAdvection must be before ablate::finiteVolume::processes::NPhaseSurfaceTension");

    nPhases = flow.GetSubDomain().GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK).numberComponents;

    // Continuous flux function
    flow.RegisterRHSFunction(PointFlux, this,
      {ablate::finiteVolume::NPhaseFlowFields::ALLAIRE}, // Calculate flux for
      {ablate::finiteVolume::NPhaseFlowFields::ALPHAK},  // Required solution fields
      {ablate::finiteVolume::NPhaseFlowFields::UI, ablate::finiteVolume::NPhaseFlowFields::AIJ});     // Required aux fields
}



/*
    This is based on a combination of
      "A conservative second order phase field model for simulation of N-phase flows" by Mirjalilia and Mani
      and the capillary (Korteweg) stress tensor.

      The capillary tensor for two-phase flow is T = sigma * (I - n n^T)| grad(c) |, where n n^T is the outer product.
      The divergence of this results in the standard surface tension force: -sigma * H * grad(c).

      Eq. (10) in Mirjalilia and Mani has the tension contribution of the i-j pair as 6 * sigma_{ij} * H_{ij} * ai * aj * grad(aij),
      where H_{ij} is the curvature of aij = ai / (ai + aj).

      Re-arranging this we get (6 * sigma_{ij} * ai * aj) * H_{ij} * grad(aij). Thus, for n-phase flow we replace this with the pairwise capillary stress tensor:
      Tij = (6 * sigma_{ij} * ai * aj) * (I - n_{ij} n^T_{ij}) | grad(aij) |
*/
PetscErrorCode ablate::finiteVolume::processes::NPhaseSurfaceTension::PointFlux(PetscInt dim, const PetscFVFaceGeom* fg,
  const PetscInt uOff[], const PetscInt uOff_x[],
  const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar field[], const PetscScalar grad[],
  const PetscInt aOff[], const PetscInt aOff_x[],
  const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar aux[], const PetscScalar gradAux[],
  PetscScalar flux[], void* ctx) {

    PetscFunctionBegin;

    auto process = (NPhaseSurfaceTension *)ctx;
    std::vector<PetscReal> sigmaij = process->sigmaij;
    const std::size_t nPhases = process->nPhases;
    const PetscReal         h = process->h;

    const PetscReal *alphak = &field[uOff[0]];
    const PetscReal    *vel = &aux[aOff[0]];
    const PetscReal   *gaij = &gradAux[aOff_x[1]];

    const PetscReal u_n = utilities::MathUtilities::DotVector(dim, vel, fg->normal);

    PetscCall(PetscArrayzero(flux, dim + 1));
    for (std::size_t i = 0; i < nPhases; ++i) {
      for (std::size_t j = i + 1; j < nPhases; ++j) {

        // Kernel times surface tension coefficient
        const PetscReal f = 6 * sigmaij[i * nPhases + j] * alphak[i] * alphak[j];

        // Gradient of A_{ij}
        const PetscReal *g = &gaij[(i * nPhases + j)*dim];

        // || grad(A_{ij}) || + h * h
        const PetscReal mag = utilities::MathUtilities::MagVector(dim, g) + h * h;

        // Gradient of A_{ij} dot face area normal
        PetscReal g_n = utilities::MathUtilities::DotVector(dim, g, fg->normal);

        // Velocity dot gradient of A_{ij}
        PetscReal g_u = utilities::MathUtilities::DotVector(dim, g, vel);

        // Energy
        flux[NPhaseFlowFields::RHOE] -= f * (mag * u_n - g_u * g_n / mag);

        // Momentum
        for (PetscInt d = 0; d < dim; ++d) flux[NPhaseFlowFields::RHOU + d] -= f * (mag * fg->normal[d] - g[d] * g_n / mag);

      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);

  }


}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process,
    ablate::finiteVolume::processes::NPhaseSurfaceTension,
    "N-phase surface tension with user-defined coefficients",
    ARG(std::vector<PetscReal>, "sigmaij", "Surface tension coefficients for each phase pair (must match number of phase pairs)"));
