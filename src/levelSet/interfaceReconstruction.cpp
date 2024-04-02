#include "interfaceReconstruction.hpp"
#include "levelSetUtilities.hpp"
#include <petsc.h>
#include <memory>
#include "LS-VOF.hpp"
#include "cellGrad.hpp"
#include "geometry.hpp"
#include "domain/range.hpp"
#include "domain/reverseRange.hpp"
#include "mathFunctions/functionWrapper.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

using namespace ablate::levelSet;




void SaveVertexDataNew(DM dm, Vec vec, const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  ablate::domain::Range range;
  const PetscReal    *array, *val;
  PetscInt      dim = subDomain->GetDimensions();
  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;
  MPI_Comm_rank(comm, &rank) >> ablate::utilities::MpiUtilities::checkError;

  ablate::domain::GetRange(dm, nullptr, 0, range);

  VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");

      for (PetscInt v = range.start; v < range.end; ++v) {
        PetscInt vert = range.points ? range.points[v] : v;
        PetscScalar *coords;

        DMPlexPointLocalFieldRead(dm, vert, field->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

        DMPlexVertexGetCoordinates(dm, 1, &vert, &coords);

        for (PetscInt d = 0; d < dim; ++d) {
          fprintf(f1, "%+.16e\t", coords[d]);
        }

        for (PetscInt i = 0; i < Nc; ++i) {
          fprintf(f1, "%+.16e\t", val[i]);
        }
        fprintf(f1, "\n");

        DMPlexVertexRestoreCoordinates(dm, 1, &vert, &coords);
      }

      fclose(f1);
    }
    MPI_Barrier(comm);
  }


  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  ablate::domain::RestoreRange(range);
}

void SaveVertexDataNew(DM dm, Vec vec, const char fname[255], const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {
  SaveVertexDataNew(dm, vec, fname, field, 1, subDomain);
}

void SaveVertexDataNew(const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  Vec           vec = subDomain->GetVec(*field);
  DM            dm  = subDomain->GetFieldDM(*field);
  SaveVertexDataNew(dm, vec, fname, field, Nc, subDomain);
}

void SaveCellDataNew(DM dm, const Vec vec, const char fname[255], const PetscInt id, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  ablate::domain::Range range;
  const PetscScalar *array;
  PetscInt      dim = subDomain->GetDimensions();
  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;
  MPI_Comm_rank(comm, &rank) >> ablate::utilities::MpiUtilities::checkError;

  subDomain->GetCellRange(nullptr, range);

  VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;



  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");

      for (PetscInt c = range.start; c < range.end; ++c) {
        PetscInt cell = range.points ? range.points[c] : c;

        if (ablate::levelSet::Utilities::ValidCell(dm, cell)) {

          PetscReal x0[3];
          DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt d = 0; d < dim; ++d) {
            fprintf(f1, "%+e\t", x0[d]);
          }

          const PetscScalar *val;
          DMPlexPointLocalFieldRead(dm, cell, id, array, &val) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt i = 0; i < Nc; ++i) {
            fprintf(f1, "%+e\t", val[i]);
          }

          fprintf(f1, "\n");
        }
      }
      fclose(f1);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
  }


  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  ablate::domain::RestoreRange(range);
}

void SaveCellDataNew(DM dm, const Vec vec, const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {
  SaveCellDataNew(dm, vec, fname, field->id, Nc, subDomain);
}




Reconstruction::Reconstruction(std::shared_ptr<ablate::domain::SubDomain> subDomainIn){

  this->subDomain = subDomainIn;


  PetscReal         h = 0.0;
  DMPlexGetMinRadius(subDomain->GetDM(), &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

  // nLevels needs to be wide enough to support the width necessary for the RBF.
  PetscInt polyAug = 2; // Looks like I need an odd augmented polynomial order for the curvature to be acceptable
  bool doesNotHaveDerivatives = false;
  bool doesNotHaveInterpolation = false;
  bool returnNeighborVertices = true;
  this->vertRBF = std::make_shared<ablate::domain::rbf::IMQ>(polyAug, 1e-2*h, doesNotHaveDerivatives, doesNotHaveInterpolation, returnNeighborVertices);
  this->vertRBF->Setup(this->subDomain);
  this->vertRBF->Initialize();




  polyAug = 2;
  returnNeighborVertices = false;
  this->cellRBF = std::make_shared<ablate::domain::rbf::MQ>(polyAug, h, doesNotHaveDerivatives, doesNotHaveInterpolation, returnNeighborVertices);
  this->cellRBF->Setup(this->subDomain);
  this->cellRBF->Initialize();

}

Reconstruction::~Reconstruction(){
  // Need to add in rbf destructors


}



/**
  * Compute the upwind derivative
  * @param dm - Domain of the gradient data.
  * @param gradArray - Array containing the cell-centered gradient
  * @param v - Vertex id
  * @param direction - The direction to be considered upwind. +1 for standard upwind, -1 of downwind
  * @param g - On input the gradient of the level-set field at a vertex. On output the upwind gradient at v
  */
void Reconstruction::VertexUpwindGrad(DM dm, PetscScalar *gradArray, const PetscInt gradID, const PetscInt v, const PetscReal direction, PetscReal *g) {
  // The upwind direction is determined using the dot product between the vector u and the vector connecting the cell-center
  //    and the vertex

  PetscInt          dim;
  PetscReal         weightTotal = 0.0;
  PetscScalar       x0[3], n[3];

  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  ablate::utilities::MathUtilities::NormVector(dim, g, n);

  DMPlexComputeCellGeometryFVM(dm, v, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

//if(fabs(x0[0]-0.3125)<0.001 && fabs(x0[1]-0.0625)<0.00){
//  printf("%ld\n", v);
//  exit(0);
//}

  for (PetscInt d = 0; d < dim; ++d) {
    g[d] = 0.0;
  }


  // Obtain all cells which use this vertex
  PetscInt nCells, *cells;
  DMPlexVertexGetCells(dm, v, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = 0; c < nCells; ++c) {
    PetscReal x[3];
    DMPlexComputeCellGeometryFVM(dm, cells[c], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

    ablate::utilities::MathUtilities::Subtract(dim, x0, x, x);
    ablate::utilities::MathUtilities::NormVector(dim, x, x);
    PetscReal dot = ablate::utilities::MathUtilities::DotVector(dim, n, x);

    dot *= direction;

    if (dot>=0.0) {

      weightTotal += dot;

      const PetscScalar *cellGrad = nullptr;
      xDMPlexPointLocalRead(dm, cells[c], gradID, gradArray, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;

      // Weighted average of the surrounding cell-center gradients.
      //  Note that technically this is (in 2D) the area of the quadrilateral that is formed by connecting
      //  the vertex, center of the neighboring edges, and the center of the triangle. As the three quadrilaterals
      //  that are formed this way all have the same area, there is no need to take into account the 1/3. Something
      //  similar should hold in 3D and for other cell types that ABLATE uses.
      for (PetscInt d = 0; d < dim; ++d) {
        g[d] += dot*cellGrad[d];
      }
    }
  }
//exit(0);
  DMPlexVertexRestoreCells(dm, v, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

  // Size of the communicator
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  int size;
  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;

  // Error checking
  if ( PetscAbs(weightTotal) < ablate::utilities::Constants::small ) {
    // When running on a single processor all vertices should have an upwind cell. Throw an error if that's not the case.
    // When running in parallel, ghost vertices at the edge of the local domain may not have any surrounding upwind cells, so
    //  ignore the error and simply set the upwind gradient to zero.
//    if ( size==1 ) {
//      throw std::runtime_error("ablate::levelSet::Utilities::VertexUpwindGrad encounted a situation where there are no upwind cells");
//    }
//    if ( size==1 ) {
//      char err[255];
//      sprintf(err, "ablate::levelSet::Utilities::VertexUpwindGrad encounted a situation where there are no upwind cells %f,%f", x0[0], x0[1]);
//      throw std::runtime_error(err);
//    }
    for (PetscInt d = 0; d < dim; ++d) {
      g[d] = 0.0;
    }
  }
  else {
    for (PetscInt d = 0; d < dim; ++d) {
      g[d] /= weightTotal;
    }
  }
}



// Inter-processor ghost cells are iterated over, so everything should work fine
void Reconstruction::CutCellLevelSetValues(std::shared_ptr<ablate::domain::SubDomain> subDomain, ablate::domain::Range cellRange, ablate::domain::Range vertRange, ablate::domain::ReverseRange reverseVertRange, const PetscInt *cellMask, DM solDM, Vec solVec, const PetscInt vofID, DM auxDM, Vec auxVec, const PetscInt normalID, const PetscInt lsID) {

  const PetscScalar *solArray = nullptr;
  PetscScalar *auxArray = nullptr;
  PetscInt *lsCount;


  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(lsCount, vertRange.end - vertRange.start) >> ablate::utilities::PetscUtilities::checkError;
  lsCount -= vertRange.start;

  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    PetscInt vert = vertRange.GetPoint(v);
    PetscReal *lsVal = nullptr;
    xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
    *lsVal = 0.0;
  }

//int rank;
//MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> ablate::utilities::MpiUtilities::checkError;

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    // Only worry about cut-cells
    if ( cellMask[c]==1 ) {
      PetscInt cell = cellRange.GetPoint(c);

      // The VOF for the cell
      const PetscScalar *vofVal = nullptr;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      // The pre-computed cell-centered normal
      const PetscScalar *n = nullptr;
      xDMPlexPointLocalRead(auxDM, cell, normalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError;

      PetscInt nv, *verts;
      DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal *lsVertVals = NULL;
      DMGetWorkArray(auxDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;

      // Level set values at the vertices
      ablate::levelSet::Utilities::VertexLevelSet_VOF(auxDM, cell, *vofVal, n, &lsVertVals);

      for (PetscInt v = 0; v < nv; ++v) {
        PetscScalar *lsVal = nullptr;
        xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
        *lsVal += lsVertVals[v];

        PetscInt vert_i = reverseVertRange.GetIndex(verts[v]);
        ++lsCount[vert_i];
      }

      DMRestoreWorkArray(auxDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexCellRestoreVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

    }
  }

  // This is no longer needed
  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    if ( lsCount[v] > 0 ) {

      PetscInt vert = vertRange.GetPoint(v);

      PetscReal *lsVal = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

      *lsVal /= lsCount[v];
    }
  }

  lsCount += vertRange.start;
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;

  VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->UpdateAuxLocalVector();

}





static PetscReal GaussianDerivativeFactor(const PetscInt dim, const PetscReal *x, const PetscReal s,  const PetscInt dx, const PetscInt dy, const PetscInt dz) {

  const PetscReal s2 = PetscSqr(s);

  const PetscInt derHash = 100*dx + 10*dy + dz;

  if (derHash > 0 && PetscAbsReal(s)<PETSC_SMALL) return (0.0);

  switch (derHash) {
    case   0: // Value
      return (1.0);
    case 100: // x
      return (x[0]/s2);
    case  10: // y
      return (x[1]/s2);
    case   1: // z
      return (x[2]/s2);
    case 200: // xx
      return ((x[0]*x[0] - s2)/PetscSqr(s2));
    case  20: // yy
      return ((x[1]*x[1] - s2)/PetscSqr(s2));
    case   2: // zz
      return ((x[2]*x[2] - s2)/PetscSqr(s2));
    case 110: // xy
      return (x[0]*x[1]/PetscSqr(s2));
    case 101: // xz
      return (x[0]*x[2]/PetscSqr(s2));
    case  11: // yz
      return (x[1]*x[2]/PetscSqr(s2));
    default:
      throw std::runtime_error("Unknown derivative request");
  }

}

// Calculate the curvature from a vertex-based level set field using Gaussian convolution.
// Right now this is just 2D for testing purposes.
void Reconstruction::CurvatureViaGaussian(DM dm, const PetscInt cell, const Vec vec, const ablate::domain::Field *lsField, double *H) {

  PetscInt dim;
  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  PetscReal h;
  DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

//  const PetscInt nQuad = 3; // Size of the 1D quadrature
//  const PetscReal quad[] = {0.0, PetscSqrtReal(3.0), -PetscSqrtReal(3.0)};
//  const PetscReal weights[] = {2.0/3.0, 1.0/6.0, 1.0/6.0};


//   Hermite-Gauss quadrature points
  const PetscInt nQuad = 4; // Size of the 1D quadrature

//   The quadrature is actually sqrt(2) times the quadrature points. This is as we are integrating
//      against the normal distribution, not exp(-x^2)
  const PetscReal quad[4] = {-0.74196378430272585764851359672636022482952014750891895361147387899499975465000530,
                             0.74196378430272585764851359672636022482952014750891895361147387899499975465000530,
                            -2.3344142183389772393175122672103621944890707102161406718291603341725665622712306,
                             2.3344142183389772393175122672103621944890707102161406718291603341725665622712306};

// The weights are the true weights divided by sqrt(pi)
  const PetscReal weights[4] = {0.45412414523193150818310700622549094933049562338805584403605771393758003145477625,
                               0.45412414523193150818310700622549094933049562338805584403605771393758003145477625,
                               0.045875854768068491816892993774509050669504376611944155963942286062419968545223748,
                               0.045875854768068491816892993774509050669504376611944155963942286062419968545223748};


  PetscReal x0[dim], vol;
  DMPlexComputeCellGeometryFVM(dm, cell, &vol, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  // Mabye relate this to PETSC_SQRT_MACHINE_EPSILON or similar?
  //  The would probably require that the derivative factor be re-done to account for round-off.
  const PetscReal sigma = h;

  std::shared_ptr<ablate::domain::rbf::RBF> vertRBF = this->vertRBF;


  PetscReal cx = 0.0, cy = 0.0, cxx = 0.0, cyy = 0.0, cxy = 0.0;

  for (PetscInt i = 0; i < nQuad; ++i) {
    for (PetscInt j = 0; j < nQuad; ++j) {

      const PetscReal dist[2] = {sigma*quad[i], sigma*quad[j]};
      PetscReal x[2] = {x0[0] + dist[0], x0[1] + dist[1]};

      const PetscReal lsVal = vertRBF->Interpolate(lsField, vec, x);

      const PetscReal wt = weights[i]*weights[j];

      cx  += wt*GaussianDerivativeFactor(dim, dist, sigma, 1, 0, 0)*lsVal;
      cy  += wt*GaussianDerivativeFactor(dim, dist, sigma, 0, 1, 0)*lsVal;
      cxx += wt*GaussianDerivativeFactor(dim, dist, sigma, 2, 0, 0)*lsVal;
      cyy += wt*GaussianDerivativeFactor(dim, dist, sigma, 0, 2, 0)*lsVal;
      cxy += wt*GaussianDerivativeFactor(dim, dist, sigma, 1, 1, 0)*lsVal;
    }
  }

  *H = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy)/PetscPowReal(cx*cx + cy*cy, 1.5);

}


//vofField: cell-based field containing the target volume-of-fluid
//lsField: vertex-based field for level set values
//normalField: cell-based vector field containing normals
//curvField: cell-based vector field containing curvature
void Reconstruction::ComputeCurvature(const Vec solVec, const ablate::domain::Field *vofField, const PetscInt nLevels, const ablate::domain::Field *lsField, const ablate::domain::Field *vertexNormalField, const ablate::domain::Field *cellNormalField, const ablate::domain::Field *curvField) {

  std::shared_ptr<ablate::domain::SubDomain> subDomain = Reconstruction::subDomain;

  // Make sure that all of the fields are in the correct locations.
  if ( vofField->location != ablate::domain::FieldLocation::SOL ){
    throw std::runtime_error("VOF field must be in SOL");
  }

  if ( lsField->location != ablate::domain::FieldLocation::AUX ){
    throw std::runtime_error("Level set field must be in AUX");
  }

  if ( vertexNormalField->location != ablate::domain::FieldLocation::AUX ){
    throw std::runtime_error("Vertex Normal field must be in AUX");
  }

  if ( cellNormalField->location != ablate::domain::FieldLocation::AUX ){
    throw std::runtime_error("Cell Normal field must be in AUX");
  }

  if ( curvField->location != ablate::domain::FieldLocation::AUX ){
    throw std::runtime_error("Curvature Field field must be in AUX");
  }


  PetscReal         h = 0.0;
  const PetscInt    dim = subDomain->GetDimensions();   // VOF and LS subdomains must have the same dimension. Can't think of a reason they wouldn't.
  PetscInt          *vertMask = nullptr, *cellMask = nullptr;
  DM                solDM = subDomain->GetDM();
  DM                auxDM = subDomain->GetAuxDM();
  Vec               auxVec = subDomain->GetAuxVector();
  const PetscScalar *solArray = nullptr;
  PetscScalar       *auxArray = nullptr;
  const PetscInt    lsID = lsField->id, vofID = vofField->id, cellNormalID = cellNormalField->id;

  DMPlexGetMinRadius(solDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size
printf("%+f\n", h);

  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;


  ablate::domain::Range cellRange, vertRange;
  subDomain->GetCellRange(nullptr, cellRange);
  subDomain->GetRange(nullptr, 0, vertRange);

  // Get the point->index mapping for cells
  ablate::domain::ReverseRange reverseVertRange = ablate::domain::ReverseRange(vertRange);
  ablate::domain::ReverseRange reverseCellRange = ablate::domain::ReverseRange(cellRange);

  // Pull some work arrays to store the mask information
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(vertMask, vertRange.end - vertRange.start);
  vertMask -= vertRange.start; // offset so that we can use start->end

  DMGetWorkArray(solDM, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(cellMask, cellRange.end - cellRange.start);
  cellMask -= cellRange.start; // offset so that we can use start->end

SaveCellDataNew(solDM, solVec, "vof.txt", vofField, 1, subDomain);

/**************** Determine the cut-cells and initial unit normal *************************************/

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    PetscInt cell = cellRange.GetPoint(c);

    if (ablate::levelSet::Utilities::ValidCell(solDM, cell)) {

      const PetscScalar *vofVal = nullptr;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      PetscScalar *n = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
      for (PetscInt d = 0; d < dim; ++d ) n[d] = 0.0;

      if ( ((*vofVal) > 0.0001) && ((*vofVal) < 0.9999) ) {

        cellMask[c] = 1;    // Mark as a cut-cell

        // Will this crap near the edges of a processor?
        if ( dim > 0 ) n[0] = cellRBF->EvalDer(solDM, solVec, vofID, cell, 1, 0, 0);
        if ( dim > 1 ) n[1] = cellRBF->EvalDer(solDM, solVec, vofID, cell, 0, 1, 0);
        if ( dim > 2 ) n[2] = cellRBF->EvalDer(solDM, solVec, vofID, cell, 0, 0, 1);

        ablate::utilities::MathUtilities::NormVector(dim, n);
        for (PetscInt d = 0; d < dim; ++d) n[d] *= -1.0;

        // Mark all vertices of this cell as associated with a cut-cell
        PetscInt nv, *verts;
        DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt v = 0; v < nv; ++v) {
          PetscInt vert_i = reverseVertRange.GetIndex(verts[v]);
          vertMask[vert_i] = 1;
        }
        DMPlexCellRestoreVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }
    }
  }

  subDomain->UpdateAuxLocalVector();



/**************** Iterate to get the level-set values at vertices *************************************/

  // Temporary level-set work array to store old values
  PetscScalar *tempLS;
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
  tempLS -= vertRange.start;

  PetscReal maxDiff = 1.0;
  PetscInt iter = 0;

  MPI_Comm auxCOMM = PetscObjectComm((PetscObject)auxDM);

SaveCellDataNew(auxDM, auxVec, "normal0.txt", cellNormalField, dim, subDomain);

  PetscReal cRange[2] = {PETSC_MAX_REAL, -PETSC_MAX_REAL};
  while ( maxDiff > 1e-3*h && iter<500 ) {

    ++iter;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v]==1) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *oldLS = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &oldLS) >> ablate::utilities::PetscUtilities::checkError;
        tempLS[v] = *oldLS;
      }
    }

    // This updates the lsField by taking the average vertex values necessary to match the VOF in cutcells
    CutCellLevelSetValues(subDomain, cellRange, vertRange, reverseVertRange, cellMask, solDM, solVec, vofID, auxDM, auxVec, cellNormalID, lsID);

    //     Update the normals
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] == 1) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscScalar *n = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
        DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
        ablate::utilities::MathUtilities::NormVector(dim, n);
      }
    }

    subDomain->UpdateAuxLocalVector();


    // Now compute the difference on this processor
    maxDiff = -1.0;
    cRange[0] = PETSC_MAX_REAL;
    cRange[1] = -PETSC_MAX_REAL;
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      if (vertMask[v] == 1) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *newLS = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &newLS) >> ablate::utilities::PetscUtilities::checkError;

        maxDiff = PetscMax(maxDiff, PetscAbsReal(tempLS[v] - *newLS));

        cRange[0] = PetscMin(cRange[0], *newLS);
        cRange[1] = PetscMax(cRange[1], *newLS);

      }
    }
    // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

    PetscPrintf(PETSC_COMM_WORLD, "Cut Cells %" PetscInt_FMT": %+e\n", iter, maxDiff) >> ablate::utilities::PetscUtilities::checkError;
  }

  if (maxDiff > 1e-3*h) {
    SaveCellDataNew(auxDM, auxVec, "normalERROR.txt", cellNormalField, dim, subDomain);
    SaveVertexDataNew(auxDM, auxVec, "ls0ERROR.txt", lsField, subDomain);
    throw std::runtime_error("Interface reconstruction has failed. Please check the number of cut-cells.\n");
  }

  cRange[0] *= -1.0;
  MPI_Allreduce(MPI_IN_PLACE, cRange, 2, MPIU_REAL, MPIU_MAX, auxCOMM);
  cRange[0] *= -1.0;



/**************** Set the data in the rest of the domain to be a large value *************************************/
PetscPrintf(PETSC_COMM_WORLD, "Setting data\n");
  // Set the vertices far away as the largest possible value in the domain with the appropriate sign.
  // This is done after the determination of cut-cells so that all vertices associated with cut-cells have been marked.
  PetscReal gMin[3], gMax[3], maxDist = -1.0;
  DMGetBoundingBox(auxDM, gMin, gMax) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    maxDist = PetscMax(maxDist, gMax[d] - gMin[d]);
  }
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    // Only worry about cells to far away
    if ( cellMask[c] == 0 && ablate::levelSet::Utilities::ValidCell(solDM, cell)) {
      const PetscScalar *vofVal = nullptr;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal sgn = PetscSignReal(0.5 - (*vofVal));

      PetscInt nv, *verts;
      DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt v = 0; v < nv; ++v) {
        PetscInt id = reverseVertRange.GetIndex(verts[v]);
        if (vertMask[id] == 0) {
          PetscScalar *lsVal = nullptr;
          xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
          *lsVal = sgn*maxDist;
        }
      }
      DMPlexCellRestoreVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }




SaveVertexDataNew(auxDM, auxVec, "ls0.txt", lsField, subDomain);




/**************** Mark the cells that need to be udpated via the reinitialization equation *************************************/
PetscPrintf(PETSC_COMM_WORLD, "Marking cells\n");
  // Mark all of the cells neighboring cells level-by-level.
  // Note that DMPlexGetNeighbors has an issue in parallel whereby cells will be missed due to the unknown partitioning -- Need to investigate
  Vec workVec, workVecGlobal;
  PetscScalar *workArray = nullptr;
  DMGetLocalVector(auxDM, &workVec);
  DMGetGlobalVector(auxDM, &workVecGlobal);

  VecZeroEntries(workVec);

  VecGetArray(workVec, &workArray);
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);
    PetscScalar *maskVal = nullptr;
    xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;
    *maskVal = cellMask[c];
  }
  VecRestoreArray(workVec, &workArray);

  DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
  DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;


  PetscReal lsRange[2] = {PETSC_MAX_REAL, -PETSC_MAX_REAL};

  for (PetscInt v = vertRange.start; v < vertRange.end; ++v){
    if (vertMask[v]==1) {
      PetscInt vert = vertRange.GetPoint(v);
      const PetscScalar *lsVal;
      xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
      lsRange[0] = PetscMin(lsRange[0], *lsVal);
      lsRange[1] = PetscMax(lsRange[1], *lsVal);
    }
  }


  lsRange[0] = -lsRange[0];
  MPI_Allreduce(MPI_IN_PLACE, lsRange, 2, MPIU_REAL, MPIU_MAX, auxCOMM);
  lsRange[0] = -lsRange[0];



int rank;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> ablate::utilities::MpiUtilities::checkError;
  for (PetscInt l = 1; l <= nLevels; ++l) {

    VecGetArray(workVec, &workArray);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);
      PetscScalar *maskVal = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;

      if ( PetscAbsScalar(*maskVal - l) < 0.1 ) {
        PetscInt nCells, *cells;
        DMPlexGetNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt i = 0; i < nCells; ++i) {
          PetscScalar *neighborMaskVal = nullptr;
          xDMPlexPointLocalRef(auxDM, cells[i], vofID, workArray, &neighborMaskVal) >> ablate::utilities::PetscUtilities::checkError;
          if ( *neighborMaskVal < 0.5 ) {
            *neighborMaskVal = l + 1;

            cellMask[reverseCellRange.GetIndex(cells[i])] = l + 1;

            PetscScalar *vofVal;
            xDMPlexPointLocalRead(solDM, cells[i], vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

            PetscInt nv, *verts;
            DMPlexCellGetVertices(auxDM, cells[i], &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

            for (PetscInt v = 0; v < nv; ++v) {
              PetscInt id = reverseVertRange.GetIndex(verts[v]);

              if (vertMask[id]==0) {
                vertMask[id] = l + 1;

                PetscScalar *lsVal;
                xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

//                PetscReal sgn = (*vofVal < 0.5 ? +1.0 : -1.0);
//                if (sgn<0) *lsVal = lsRange[0];
//                else *lsVal = lsRange[1];

                *lsVal = lsRange[ *vofVal < 0.5 ? 1 : 0 ];


//                *lsVal = 2*h*sgn;
//                if (sgn<0) {
//                  *lsVal = lsRange[0] - (l-0.5)*h;
//                }
//                else {
//                  *lsVal = lsRange[1] + (l-0.5)*h;
//                }
//                *lsVal = sgn*l*h;
              }
            }

            DMPlexCellRestoreVertices(auxDM, cells[i], &nv, &verts);
          }
        }
        DMPlexRestoreNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      }
    }
    VecRestoreArray(workVec, &workArray);

    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;
  }

  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->UpdateAuxLocalVector();
SaveVertexDataNew(auxDM, auxVec, "ls1.txt", lsField, subDomain);

//exit(0);
PetscPrintf(PETSC_COMM_WORLD, "Reinit\n");
/**************** Level-set reinitialization equation *************************************/
  const PetscInt vertexNormalID = vertexNormalField->id;
  const PetscInt curvID = curvField->id;



  maxDiff = 1.0;
  iter = 0;

  while (maxDiff>1.e-2 && iter<(nLevels*5)) {
    ++iter;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 0) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *phi = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
        tempLS[v] = *phi;
      }
    }


    // Determine the current gradient at cells that need updating
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] > 0) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscReal *g = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, g) >> ablate::utilities::PetscUtilities::checkError;
      }
    }
    subDomain->UpdateAuxLocalVector();

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 0) {
        PetscInt vert = vertRange.GetPoint(v);

        PetscReal *g = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, g) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

SaveVertexDataNew(auxDM, auxVec, "phi.txt", lsField, 1, subDomain);
SaveCellDataNew(auxDM, auxVec, "cGrad.txt", cellNormalField, dim, subDomain);
SaveVertexDataNew(auxDM, auxVec, "vGrad.txt", vertexNormalField, dim, subDomain);
//exit(0);
    maxDiff = -PETSC_MAX_REAL;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      if (vertMask[v] > 1) {
        PetscInt vert = vertRange.GetPoint(v);

        PetscReal *phi = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;

        const PetscReal *arrayG = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &arrayG) >> ablate::utilities::PetscUtilities::checkError;

        PetscReal g[dim];
        for (PetscInt d = 0; d < dim; ++d){
          g[d] = arrayG[d];
        }

//        VertexUpwindGrad(auxDM, auxArray, cellNormalID, vert, PetscSignReal(*phi), g);

        PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);

        PetscReal sgn = (tempLS[v] > 0.0 ? +1.0 : -1.0);

//          PetscReal sgn = (*phi)/PetscSqrtReal(PetscSqr(*phi) + PetscSqr(h));

        *phi = tempLS[v] - 0.5*h*sgn*(nrm - 1.0);
        maxDiff = PetscMax(maxDiff, PetscAbsReal(nrm - 1.0));
      }
    }

    subDomain->UpdateAuxLocalVector();



     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);



    PetscPrintf(PETSC_COMM_WORLD, "Reinit %3" PetscInt_FMT": %e\n", iter, maxDiff);

    char fname[255];
    sprintf(fname, "phi%03ld.txt", iter);
    SaveVertexDataNew(auxDM, auxVec, fname, lsField, 1, subDomain);

  }

SaveVertexDataNew(auxDM, auxVec, "ls2.txt", lsField, subDomain);
printf("1617\n");


  // Calculate unit normal vector based on the updated level set values at the vertices
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    if (cellMask[c] > 0) {
      PetscInt cell = cellRange.GetPoint(c);
      PetscScalar *n = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
      DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
      ablate::utilities::MathUtilities::NormVector(dim, n);
    }
  }

  // Calculate vertex-based unit normal vector based on the updated level set values at the vertices
  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

    if (vertMask[v] > 0) {
      PetscInt vert = vertRange.GetPoint(v);

      PetscScalar *n = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &n);
      DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
      ablate::utilities::MathUtilities::NormVector(dim, n);
    }
  }

  subDomain->UpdateAuxLocalVector();



//for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

//  if (vertMask[v] > 0) {
//    PetscInt vert = vertRange.GetPoint(v);
//    PetscScalar *n = nullptr;
//    xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &n);
//    *n = 1.0;
//  }
//}


  // Curvature
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);
    PetscScalar *H = nullptr;
    xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &H);

    if (cellMask[c] == 1 ) {
      CurvatureViaGaussian(auxDM, cell, auxVec, lsField, H);
    }
    else {
      *H = 0.0;
    }
  }
  subDomain->UpdateAuxLocalVector();
SaveCellDataNew(auxDM, auxVec, "curvGaussian.txt", curvField, 1, subDomain);


  // Curvature
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    PetscInt cell = cellRange.GetPoint(c);
    PetscScalar *H = nullptr;
    xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &H);

    if (cellMask[c] == 1 ) {
      *H = ablate::levelSet::geometry::Curvature(vertRBF, lsField, cell);
    }
    else {
      *H = 0.0;
    }

  }
  subDomain->UpdateAuxLocalVector();
SaveCellDataNew(auxDM, auxVec, "curvRBF.txt", curvField, 1, subDomain);


  // Curvature
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    PetscInt cell = cellRange.GetPoint(c);
    PetscScalar *H = nullptr;
    xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &H);

    if (cellMask[c] == 1 ) {
      *H = 0;
      for (PetscInt d = 0; d < dim; ++d) {
        PetscReal g[dim];
        DMPlexCellGradFromVertex(auxDM, cell, auxVec, vertexNormalID, d, g) >> ablate::utilities::PetscUtilities::checkError;
        *H += g[d];
      }
    }
    else {
      *H = 0.0;
    }

  }
  subDomain->UpdateAuxLocalVector();
SaveCellDataNew(auxDM, auxVec, "curvDiv.txt", curvField, 1, subDomain);


exit(0);
  // Extension
  PetscInt vertexCurvID = lsID; // Store the vertex curvatures in the work vec at the same location as the level-set



#if 0 // Smoothing iteration
  for (PetscInt iter = 0; iter < 10; ++iter) {

    VecGetArray(workVec, &workArray);
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      PetscInt vert = vertRange.GetPoint(v);
      PetscReal *vertexH = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &vertexH);

      if (vertMask[v] > 0) {

  //      for (PetscInt d = 0; d < dim; ++d) {
  //        PetscReal g[dim];
  //        DMPlexVertexGradFromCell(auxDM, vert, auxVec, cellNormalID, d, g);
  //        *vertexH += g[d];
  //      }


          PetscInt nc = 0;
          *vertexH = 0.0;

          PetscInt nCells, *cells;
          DMPlexVertexGetCells(auxDM, vert, &nCells, &cells);
          for (PetscInt c = 0; c < nCells; ++c) {

            PetscInt id = reverseCellRange.GetIndex(cells[c]);

            if (cellMask[id] > 0) {
              ++nc;
              const PetscReal *cellH = nullptr;
              xDMPlexPointLocalRead(auxDM, cells[c], curvID, auxArray, &cellH);

              *vertexH += *cellH;
            }
          }

          *vertexH /= nc;

          DMPlexVertexRestoreCells(auxDM, v, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;


      }
      else {
        *vertexH = 0.0;
      }
    }
    VecRestoreArray(workVec, &workArray);

    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;

    VecGetArray(workVec, &workArray);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

      PetscInt cell = cellRange.GetPoint(c);

      if (cellMask[c] > 0) {

        PetscScalar *H = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &H);

        PetscInt nVerts, *verts;
        DMPlexCellGetVertices(auxDM, cell, &nVerts, &verts) >> ablate::utilities::PetscUtilities::checkError;

        *H = 0.0;

        PetscInt nv = 0;
        for (PetscInt v = 0; v < nVerts; ++v) {

          PetscInt id = reverseVertRange.GetIndex(verts[v]);

          if (vertMask[id] > 0) {

            PetscScalar *vH = nullptr;
            xDMPlexPointLocalRef(auxDM, verts[v], vertexCurvID, workArray, &vH);

            *H += *vH;
            ++nv;
          }

        }
        *H /= nv;


        DMPlexCellRestoreVertices(auxDM, cell, &nVerts, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }
    }
    VecRestoreArray(workVec, &workArray);

    subDomain->UpdateAuxLocalVector();

  } // End smoothing iteration
#endif


  DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
  DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;


  SaveVertexDataNew(auxDM, workVec, "vertexH0.txt", lsField, subDomain);
//printf("1787\n");
//exit(0);

  maxDiff = PETSC_MAX_REAL;
  iter = 0;
  while ( maxDiff>5e-2 && iter<500) {
    ++iter;

    VecGetArray(workVec, &workArray);

    // Curvature gradient at the cell-center
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] > 0) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscScalar *g = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, workArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, workVec, vertexCurvID, 0, g);
      }
    }

    PetscReal oldMaxDiff = maxDiff;

    maxDiff = -PETSC_MAX_REAL;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 0) {
        PetscInt vert = vertRange.GetPoint(v);
        PetscReal g[dim];
        const PetscReal *phi = nullptr, *n = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, vert, vertexNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt d = 0; d < dim; ++d) g[d] = n[d];

        VertexUpwindGrad(auxDM, workArray, cellNormalID, vert, PetscSignReal(*phi), g);

        PetscReal dH = 0.0;
        for (PetscInt d = 0; d < dim; ++d) dH += g[d]*n[d];


        PetscReal *H = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &H);

        PetscReal s = *phi/PetscSqrtReal(PetscSqr(*phi) + h*h);

        *H -= 0.5*h*s*dH;
//        *H = PetscMax(*H, -1.0/h);
//        *H = PetscMin(*H,  1.0/h);


        PetscReal *mag = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, workArray, &mag) >> ablate::utilities::PetscUtilities::checkError;
        *mag = PetscAbsReal(dH);
      }
    }
    VecRestoreArray(workVec, &workArray);

    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;



    // This is temporary until after the review.
    // The norm magnitude is incorrect at the edge of processor domains. There needs to be a way to identify
    //  cell which are ghost cells as they will have incorrect answers.
    VecGetArray(workVec, &workArray);
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 0) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *mag = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, vertexNormalID, workArray, &mag) >> ablate::utilities::PetscUtilities::checkError;
        maxDiff = PetscMax(maxDiff, PetscAbsReal(*mag));
      }
    }

    VecRestoreArray(workVec, &workArray);


     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

    PetscPrintf(PETSC_COMM_WORLD, "Extension %3" PetscInt_FMT": %e\n", iter, maxDiff);


    if ((maxDiff > oldMaxDiff) && (maxDiff<1e-1)) iter = PETSC_INT_MAX;


  }

  SaveVertexDataNew(auxDM, workVec, "vertexCurv.txt", lsField, subDomain);


  // Now set the curvature at the cell-center via averaging
  VecGetArray(workVec, &workArray);
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    if (cellMask[c] > 0) {
      PetscInt cell = cellRange.GetPoint(c);

//      PetscScalar *n = nullptr;
//      xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
//      if ( dim > 0 ) n[0] = rbf->EvalDer(auxDM, auxVec, lsID, cell, 1, 0, 0);
//      if ( dim > 1 ) n[1] = rbf->EvalDer(auxDM, auxVec, lsID, cell, 0, 1, 0);
//      if ( dim > 2 ) n = rbf->EvalDer(auxDM, auxVec, lsID, cell, 0, 0, 1);
//      ablate::utilities::MathUtilities::NormVector(dim, n);


      PetscScalar *cellH = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &cellH) >> utilities::PetscUtilities::checkError;

      *cellH = 0.0;

      PetscInt nv, *verts;
      DMPlexCellGetVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      for (PetscInt i = 0; i < nv; ++i) {
        const PetscReal *H;
        xDMPlexPointLocalRead(auxDM, verts[i], vertexCurvID, workArray, &H) >> utilities::PetscUtilities::checkError;
        *cellH += *H;
      }
      *cellH /= nv;
//      *cellH = PetscMax(*cellH, -1.0/h);
//      *cellH = PetscMin(*cellH,  1.0/h);

//*cellH = 5.0*tanh(*cellH/5.0);

//*cellH = 1.0;

      DMPlexCellRestoreVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }
  VecRestoreArray(workVec, &workArray);

  subDomain->UpdateAuxLocalVector();


  SaveCellDataNew(auxDM, auxVec, "H.txt", curvField, 1, subDomain);
printf("1960\n");
exit(0);
  SaveVertexDataNew(auxDM, workVec, "vertexH1.txt", lsField, subDomain);
  DMRestoreLocalVector(auxDM, &workVec) >> utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(auxDM, &workVecGlobal) >> utilities::PetscUtilities::checkError;



  // Cleanup all memory
//  closestCell += vertRange.start; // offset so that we can use start->end
//  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &closestCell) >> ablate::utilities::PetscUtilities::checkError;
  tempLS += vertRange.start;
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
  vertMask += vertRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  cellMask += cellRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
  DMRestoreWorkArray(solDM, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->RestoreRange(vertRange);

  VecRestoreArray(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;


}
