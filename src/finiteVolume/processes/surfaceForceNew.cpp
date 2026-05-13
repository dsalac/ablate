#include "surfaceForceNew.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "finiteVolume/stencils/gaussianConvolution.hpp"


ablate::finiteVolume::processes::SurfaceForceNew::SurfaceForceNew(PetscReal sigma) : sigma(sigma) {}

ablate::finiteVolume::processes::SurfaceForceNew::~SurfaceForceNew() {}

void ablate::finiteVolume::processes::SurfaceForceNew::ClearData() {
  if (cellGaussianConv) cellGaussianConv->~GaussianConvolution();
  if (subDM) DMDestroy(&subDM);
  if (subIS) ISDestroy(&subIS);
}

void ablate::finiteVolume::processes::SurfaceForceNew::GetFieldVectors(const ablate::domain::SubDomain& subDomain, Vec *subLocalVec, Vec *subGlobalVec) {

    // A copy is made so that the VOF values can be compared when updating the other fields

    const ablate::domain::Field &vofField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    Vec entireVec = subDomain.GetVec(vofField);

    DMGetLocalVector(subDM, subLocalVec) >> ablate::utilities::PetscUtilities::checkError;
    DMGetGlobalVector(subDM, subGlobalVec) >> ablate::utilities::PetscUtilities::checkError;

    if (vofField.location == ablate::domain::FieldLocation::SOL) {

      // Copy the data to the global vec
      VecISCopy(entireVec, subIS, SCATTER_REVERSE, *subGlobalVec) >> ablate::utilities::PetscUtilities::checkError;

      // Populate the local vector
      DMGlobalToLocal(subDM, *subGlobalVec, INSERT_VALUES, *subLocalVec) >> ablate::utilities::PetscUtilities::checkError;

    } else if (vofField.location == ablate::domain::FieldLocation::AUX) {

      // Copy the data to the local vec
      VecISCopy(entireVec, subIS, SCATTER_REVERSE, *subLocalVec) >> ablate::utilities::PetscUtilities::checkError;

      // Populate the global vector
      DMLocalToGlobal(subDM, *subLocalVec, INSERT_VALUES, *subGlobalVec) >> ablate::utilities::PetscUtilities::checkError;
    } else {
      throw std::invalid_argument("Volume fraction field is not contained in either the SOL or AUX vecs!");
    }

}

// Run once per simulation
void ablate::finiteVolume::processes::SurfaceForceNew::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

  // List of required fields and locations
  std::string fieldList[] = {
                              ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD,
                              ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD
                            };

  ablate::domain::FieldLocation locationList[] = {
                              ablate::domain::FieldLocation::SOL,
                              ablate::domain::FieldLocation::SOL
                            };

  ablate::domain::SubDomain& subDomain = flow.GetSubDomain();
  PetscInt i = 0;
  for (auto fieldName : fieldList) {
    if (!(subDomain.ContainsField(fieldName))) {
      throw std::runtime_error("ablate::finiteVolume::processes::SurfaceForceNew expects a "+ fieldName +" field to be defined.");
    }
    const ablate::domain::Field field = subDomain.GetField(fieldName);
    if (field.location != locationList[i++]) {
      throw std::runtime_error("ablate::finiteVolume::processes::SurfaceForceNew: "+ fieldName +" is in the incorrect location.");
    }
  }

  flow.RegisterRHSFunction(ComputeSource, this);

}

// Every time the mesh changes
void ablate::finiteVolume::processes::SurfaceForceNew::Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) {


  ablate::domain::SubDomain& subDomain = flow.GetSubDomain();

  ClearData();

  // Get the DM that contains JUST the vof field. This will be duplicated so that we can use DMGetLocalVector, which is
  // orders-or-magnitude faster than anything else for repeated calls.
  const ablate::domain::Field vofField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
  DM entireDM = subDomain.GetFieldDM(vofField);
  DMCreateSubDM(entireDM, 1, &vofField.id, &subIS, &subDM) >> ablate::utilities::PetscUtilities::checkError;

  // The geometry vectors must be attached to the DM before DMPlexPointGeometricData is called.
  //  It this isn't done the code may hang when run in parallel as DMPlexComputeGeometryFVM must be run by all ranks.
  //  When using DMPlexPointGeometricData not all ranks may reach it and it will hang.
  Vec cellGeomVec, faceGeomVec;
  DMPlexComputeGeometryFVM(subDM, &cellGeomVec, &faceGeomVec) >> ablate::utilities::PetscUtilities::checkError;
  PetscObjectCompose((PetscObject)subDM, "DMPlex_cellgeom_fvm", (PetscObject)cellGeomVec) >> ablate::utilities::PetscUtilities::checkError;
  PetscObjectCompose((PetscObject)subDM, "DMPlex_facegeom_fvm", (PetscObject)faceGeomVec) >> ablate::utilities::PetscUtilities::checkError;
  VecDestroy(&cellGeomVec) >> ablate::utilities::PetscUtilities::checkError;
  VecDestroy(&faceGeomVec) >> ablate::utilities::PetscUtilities::checkError;

  if (cellRange.is) flow.RestoreRange(cellRange);
  flow.GetCellRangeWithoutGhost(cellRange);

  reverseCellRange = ablate::domain::ReverseRange(cellRange);

  // Using cell-center data compute the gaussian convolution at a cell
  PetscInt dim = subDomain.GetDimensions();
  cellGaussianConv = std::make_shared<ablate::finiteVolume::stencil::GaussianConvolution>(subDM, 0.5, dim, dim);

  cellGaussianConv->FormAllLists();

}

PetscErrorCode ablate::finiteVolume::processes::SurfaceForceNew::ComputeSource(const FiniteVolumeSolver &fvSolver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {

  PetscFunctionBegin;

  auto surfaceForceProcess = (ablate::finiteVolume::processes::SurfaceForceNew *)ctx;
  const ablate::domain::SubDomain& subDomain = fvSolver.GetSubDomain();
  const ablate::domain::Field vofField = subDomain.GetField(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
  DM subDM = surfaceForceProcess->subDM;
  Vec vofVecs[2] = {nullptr, nullptr};
  PetscScalar *vofArrays[2] = {nullptr, nullptr};
  ablate::domain::Range cellRange = surfaceForceProcess->cellRange;



  surfaceForceProcess->GetFieldVectors(subDomain, &vofVecs[LOCAL], &vofVecs[GLOBAL]);
  PetscCall(VecGetArray(vofVecs[LOCAL], &vofArrays[LOCAL]));
  PetscCall(VecGetArray(vofVecs[GLOBAL], &vofArrays[GLOBAL]));

  FILE *f1 = fopen("vof.txt", "w");
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    const PetscInt cell = cellRange.GetPoint(c);
    PetscInt dx[3] = {0, 0, 0};
    PetscReal cg, g[2];
    surfaceForceProcess->cellGaussianConv->Evaluate(cell, dx, subDM, -1, vofArrays[LOCAL], 0, 1, &cg);

    dx[0] = 1;
    surfaceForceProcess->cellGaussianConv->Evaluate(cell, dx, subDM, -1, vofArrays[LOCAL], 0, 1, &g[0]);
    dx[0] = 0; dx[1] = 1;
    surfaceForceProcess->cellGaussianConv->Evaluate(cell, dx, subDM, -1, vofArrays[LOCAL], 0, 1, &g[1]);

    PetscReal x[3];
    const PetscScalar *g0;
    DMPlexPointLocalRead(subDM, cell, vofArrays[LOCAL], &g0) >> ablate::utilities::PetscUtilities::checkError;
    DMPlexComputeCellGeometryFVM(subDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
    fprintf(f1, "%+e\t%+e\t%+e\t%+e\t%+e\t%+e\n", x[0], x[1], g0[0], cg, g[0], g[1]);


  }
  fclose(f1);
  printf("%s::%d\n", __FILE__, __LINE__);
  exit(0);


  PetscCall(VecRestoreArray(vofVecs[LOCAL], &vofArrays[LOCAL]));
  PetscCall(VecRestoreArray(vofVecs[GLOBAL], &vofArrays[GLOBAL]));
  PetscCall(DMRestoreGlobalVector(subDM, &vofVecs[GLOBAL]));
  PetscCall(DMRestoreLocalVector(subDM, &vofVecs[LOCAL]));


  PetscFunctionReturn(PETSC_SUCCESS);


}


REGISTER(ablate::finiteVolume::processes::Process,
         ablate::finiteVolume::processes::SurfaceForceNew,
         "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient")
);

