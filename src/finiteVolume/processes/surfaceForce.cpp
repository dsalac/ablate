#include "domain/RBF/phs.hpp"
#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "levelSet/levelSetUtilities.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "mathFunctions/functionFactory.hpp"








#define xexit(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr, \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n", \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}
ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma) : sigma(sigma) {
    printf("Sigma is equal to %e\n", sigma);
}

// Done once at the beginning of every run
void ablate::finiteVolume::processes::SurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    flow.RegisterRHSFunction(ComputeSource, this);
}

// Called every time the mesh changes
void ablate::finiteVolume::processes::SurfaceForce::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
  SurfaceForce::subDomain = solver.GetSubDomainPtr();

//  SurfaceForce::reconstruction = std::make_shared<ablate::levelSet::Reconstruction>(subDomain);

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
    PetscScalar *auxArray = nullptr;
    const PetscScalar *xArray = nullptr;

    // The grid spacing
    PetscReal h;
    DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
    h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

    ablate::levelSet::Utilities::Reinitialize(flow, subDomain, locX, vofField, 8, lsField, vertexNormalField, cellNormalField, curvField);

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();


    VecGetArray(locF, &fArray) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;
    VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
    const ablate::domain::Field *tensionForceField = &(subDomain->GetField("surfaceTensionForce"));

    flow.GetCellRangeWithoutGhost(cellRange);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);

      PetscScalar *force = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, tensionForceField->id, auxArray, &force) >> utilities::PetscUtilities::checkError;
      for (PetscInt d = 0; d < dim; ++d) force[d] = 0.0;


      if (ablate::levelSet::Utilities::ValidCell(auxDM, cell)) {

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
        dirac = SmoothDirac(cellPhi, 0.0, 2.0*h);



        if (dirac > 1e-10){
          // Normal at the cell-center
          PetscReal *n = nullptr;
          xDMPlexPointLocalRead(auxDM, cell, cellNormalField->id, auxArray, &n);

          // Curvature at the cell-center
          PetscReal *H = nullptr;
          xDMPlexPointLocalRead(auxDM, cell, curvField->id, auxArray, &H);

          const PetscScalar *euler = nullptr;
          xDMPlexPointLocalRead(eulerDM, cell, eulerField->id, xArray, &euler) >> utilities::PetscUtilities::checkError;
          const PetscScalar density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

          PetscScalar *eulerSource = nullptr;
          xDMPlexPointLocalRef(eulerDM, cell, eulerField->id, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;

          for (PetscInt d = 0; d < dim; ++d) {
              // calculate surface force and energy

              PetscReal surfaceForce = - dirac* density * sigma * H[0] * n[d];
              PetscReal vel = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
              PetscReal surfaceEnergy = surfaceForce * vel;

              // add in the contributions
              eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] += surfaceForce;
              eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += surfaceEnergy;

              force[d] = surfaceForce;

          }
        }
      }


    }
    flow.RestoreRange(cellRange);

    // Cleanup
    VecRestoreArray(locF, &fArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

    PetscFunctionReturn(PETSC_SUCCESS);
}

ablate::finiteVolume::processes::SurfaceForce::~SurfaceForce() {

}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));
