#include "domain/RBF/phs.hpp"
#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "levelSet/levelSetUtilities.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "mathFunctions/functionFactory.hpp"

ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma) : sigma(sigma) {}

// Done once at the beginning of every run
void ablate::finiteVolume::processes::SurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    flow.RegisterRHSFunction(ComputeSource, this);
}

// Called every time the mesh changes
void ablate::finiteVolume::processes::SurfaceForce::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
  SurfaceForce::subDomain = solver.GetSubDomainPtr();
}


inline PetscReal SmoothDirac(PetscReal c, PetscReal c0, PetscReal t) {
  return (PetscAbsReal(c-c0) < t ? 0.5*(1.0 + cos(M_PI*(c - c0)/t))/t : 0);
}

PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const ablate::finiteVolume::FiniteVolumeSolver &flow, DM dm, PetscReal time, Vec locX, Vec locF, void *ctx) {
    PetscFunctionBegin;

    ablate::finiteVolume::processes::SurfaceForce *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    const PetscInt dim = subDomain->GetDimensions();
    ablate::domain::Range cellRange;

    const ablate::domain::Field *vofField = &(subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD));
    const ablate::domain::Field *lsField = &(subDomain->GetField("levelSet"));
    const ablate::domain::Field *vertexNormalField = &(subDomain->GetField("vertexNormal"));
    const ablate::domain::Field *curvField = &(subDomain->GetField("curvature"));
    const ablate::domain::Field *cellNormalField = &(subDomain->GetField("cellNormal"));
    const ablate::domain::Field *eulerField = &(subDomain->GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD));

    DM eulerDM = subDomain->GetFieldDM(*eulerField); // Get an euler-specific DM in case it's not in the same solution vector as the VOF field
    const PetscReal sigma = process->sigma; // Surface tension coefficient

    PetscScalar *fArray = nullptr;
    const PetscScalar *auxArray = nullptr, *xArray = nullptr;

    // The grid spacing
    PetscReal h;
    DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
    h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

    ablate::levelSet::Utilities::Reinitialize(flow, subDomain, locX, vofField, 8, lsField, vertexNormalField, cellNormalField, curvField);

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();


    VecGetArray(locF, &fArray) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

    flow.GetCellRangeWithoutGhost(cellRange);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);

      PetscReal cellPhi = 0.0, dirac = -1.0;

      PetscInt nv, *verts;
      DMPlexCellGetVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      for (PetscInt v = 0; v < nv; ++v) {
        const PetscReal *phi = nullptr;
        xDMPlexPointLocalRead(auxDM, verts[v], lsField->id, auxArray, &phi);

        cellPhi += *phi;
      }
      cellPhi /= nv;
      DMPlexCellRestoreVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      dirac = SmoothDirac(cellPhi, 0.0, 1.5*h);

      PetscScalar *eulerSource = nullptr;
      xDMPlexPointLocalRef(eulerDM, cell, eulerField->id, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;

      // Start by zeroing out everything
      eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHO] = 0.0;
      eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = 0.0;
      for (PetscInt d = 0; d < dim; ++d) {
          eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0.0;
      }

      if (dirac > 1e-12){
        // Normal at the cell-center
        PetscReal *n = nullptr;
        xDMPlexPointLocalRead(auxDM, cell, cellNormalField->id, auxArray, &n);

        // Curvature at the cell-center
        PetscReal *H = nullptr;
        xDMPlexPointLocalRead(auxDM, cell, curvField->id, auxArray, &H);

        const PetscScalar *euler = nullptr;
        xDMPlexPointLocalRead(eulerDM, cell, eulerField->id, xArray, &euler) >> utilities::PetscUtilities::checkError;
        const PetscScalar density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

        for (PetscInt d = 0; d < dim; ++d) {
            // calculate surface force and energy

            PetscReal surfaceForce = -dirac* density * sigma * H[0] * n[d];
            PetscReal vel = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
            PetscReal surfaceEnergy = surfaceForce * vel;

            // add in the contributions
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = surfaceForce;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += surfaceEnergy;
        }
      }


    }
    flow.RestoreRange(cellRange);

    // Cleanup
    VecRestoreArray(locF, &fArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

    PetscFunctionReturn(PETSC_SUCCESS);
}

ablate::finiteVolume::processes::SurfaceForce::~SurfaceForce() {

}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));
