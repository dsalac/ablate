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

void ablate::levelSet::Utilities::CellValGrad(DM dm, const PetscInt p, PetscReal *c, PetscReal *c0, PetscReal *g) {
    DMPolytopeType ct;
    PetscInt Nc;
    PetscReal *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;
    PetscReal x0[3];

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Center of the cell
    DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    // Get the cell type and call appropriate VOF function
    DMPlexGetCellType(dm, p, &ct) >> ablate::utilities::PetscUtilities::checkError;
    switch (ct) {
        case DM_POLYTOPE_SEGMENT:
            Grad_1D(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_TRIANGLE:
            Grad_2D_Tri(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_QUADRILATERAL:
            Grad_2D_Quad(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_TETRAHEDRON:
            Grad_3D_Tetra(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_HEXAHEDRON:
            Grad_3D_Hex(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        default:
            throw std::invalid_argument("No element geometry for cell " + std::to_string(p) + " with type " + DMPolytopeTypes[ct]);
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;
}

void ablate::levelSet::Utilities::CellValGrad(DM dm, const PetscInt fid, const PetscInt p, Vec f, PetscReal *c0, PetscReal *g) {
    PetscInt nv, *verts;
    const PetscScalar *fvals, *v;
    PetscScalar *c;

    DMPlexCellGetVertices(dm, p, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

    DMGetWorkArray(dm, nv, MPIU_SCALAR, &c) >> ablate::utilities::PetscUtilities::checkError;

    VecGetArrayRead(f, &fvals) >> utilities::PetscUtilities::checkError;

    for (PetscInt i = 0; i < nv; ++i) {
        // DMPlexPointLocalFieldRead isn't behaving like I would expect. If I don't make f a pointer then it just returns zero.
        //    Additionally, it looks like it allows for the editing of the value.
        if (fid >= 0) {
            DMPlexPointLocalFieldRead(dm, verts[i], fid, fvals, &v) >> utilities::PetscUtilities::checkError;
        } else {
            DMPlexPointLocalRead(dm, verts[i], fvals, &v) >> utilities::PetscUtilities::checkError;
        }

        c[i] = *v;
    }

    DMPlexCellRestoreVertices(dm, p, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

    ablate::levelSet::Utilities::CellValGrad(dm, p, c, c0, g);

    DMRestoreWorkArray(dm, nv, MPIU_SCALAR, &c) >> ablate::utilities::PetscUtilities::checkError;
}

void ablate::levelSet::Utilities::CellValGrad(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *field, const PetscInt p, PetscReal *c0, PetscReal *g) {
    DM dm = subDomain->GetFieldDM(*field);
    Vec f = subDomain->GetVec(*field);
    ablate::levelSet::Utilities::CellValGrad(dm, field->id, p, f, c0, g);
}

void ablate::levelSet::Utilities::VertexToVertexGrad(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *field, const PetscInt p, PetscReal *g) {
    // Given a field determine the gradient at a vertex

    DM dm = subDomain->GetFieldDM(*field);
    Vec vec = subDomain->GetVec(*field);

    DMPlexVertexGradFromVertex(dm, p, vec, field->id, 0, g) >> ablate::utilities::PetscUtilities::checkError;
}


void DMPlexVertexDivFromCellUpwind(DM dm, const PetscInt v, Vec data, const PetscInt fID, const PetscReal s, const PetscReal g[], PetscReal *div) {

    const PetscScalar *dataArray;
    PetscInt cStart, cEnd;
    PetscInt dim;
    PetscInt nStar, *star = NULL;

    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    PetscReal x0[dim];
    DMPlexComputeCellGeometryFVM(dm, v, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArrayRead(data, &dataArray) >> ablate::utilities::PetscUtilities::checkError;

    *div = 0.0;
    PetscReal totalVol = 0.0;

    // Everything using this vertex
    DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &nStar, &star) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt st = 0; st < nStar * 2; st += 2) {
        if (star[st] >= cStart && star[st] < cEnd) {  // It's a cell

            PetscReal x[dim];
            DMPlexComputeCellGeometryFVM(dm, star[st], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

            PetscReal dot = 0.0;
            for (PetscInt d = 0; d < dim; ++d) {
              dot += g[d]*(x0[d] - x[d]);
            }

            if (s*dot>=0.0) {
              // Surface area normal
              PetscScalar N[3];
              DMPlexCornerSurfaceAreaNormal(dm, v, star[st], N) >> ablate::utilities::PetscUtilities::checkError;

              const PetscScalar *val;
              xDMPlexPointLocalRead(dm, star[st], fID, dataArray, &val) >> ablate::utilities::PetscUtilities::checkError;

              for (PetscInt d = 0; d < dim; ++d) *div += val[d]*N[d];

              totalVol += ablate::utilities::MathUtilities::MagVector(dim, N);
            }
        }
    }
    DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &nStar, &star) >> ablate::utilities::PetscUtilities::checkError;

    VecRestoreArrayRead(data, &dataArray) >> ablate::utilities::PetscUtilities::checkError;

    if (totalVol > 0.0) *div /= totalVol;

}

/**
  * Compute the upwind derivative
  * @param dm - Domain of the gradient data.
  * @param gradArray - Array containing the cell-centered gradient
  * @param v - Vertex id
  * @param direction - The direction to be considered upwind. +1 for standard upwind, -1 of downwind
  * @param g - On input the gradient of the level-set field at a vertex. On output the upwind gradient at v
  */
static void VertexUpwindGrad(DM dm, PetscScalar *gradArray, const PetscInt gradID, const PetscInt v, const PetscReal direction, PetscReal *g) {
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

// Given a level set and normal at the cell center compute the level set values at the vertices assuming a straight interface
void ablate::levelSet::Utilities::VertexLevelSet_LS(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *n, PetscReal **c) {
    PetscInt dim, Nc, nVerts, i, j;
    PetscReal x0[3] = {0.0, 0.0, 0.0};
    PetscReal *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;

    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    // The cell center
    DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Number of vertices
    nVerts = Nc / dim;

    if (*c == NULL) {
        PetscMalloc1(nVerts, c) >> ablate::utilities::PetscUtilities::checkError;
    }

    // The level set value of each vertex. This assumes that the interface is a line/plane
    //    with the given unit normal.
    for (i = 0; i < nVerts; ++i) {
        (*c)[i] = c0;
        for (j = 0; j < dim; ++j) {
            (*c)[i] += n[j] * (coords[i * dim + j] - x0[j]);
        }
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;



}

// Given a cell VOF and normal at the cell center compute the level set values at the vertices assuming a straight interface
void ablate::levelSet::Utilities::VertexLevelSet_VOF(DM dm, const PetscInt p, const PetscReal targetVOF, const PetscReal *n, PetscReal **c) {
    PetscReal vof;         // current VOF of the cell
    PetscReal area;        // length (2D) or area (3D) of the cell face
    PetscReal cellVolume;  // Area (2D) or volume (3D) of the cell
    const PetscReal tol = 1e-8;
    PetscInt i;
    PetscReal offset;
    PetscReal vofError;
    PetscInt nv;

    // Get the number of vertices for the cell
    DMPlexCellGetNumVertices(dm, p, &nv) >> ablate::utilities::PetscUtilities::checkError;

    // Get an initial guess at the vertex level set values assuming that the interface passes through the cell-center.
    // Also allocates c if c==NULL on entry
    ablate::levelSet::Utilities::VertexLevelSet_LS(dm, p, 0.0, n, c);

    // Get the resulting VOF from the initial guess
    ablate::levelSet::Utilities::VOF(dm, p, *c, &vof, &area, &cellVolume);
    vofError = targetVOF - vof;

    while (fabs(vofError) > tol) {
        // The amount the center level set value needs to shift by.
        offset = vofError * cellVolume / area;

        // If this isn't damped then it will overshoot and there will be no interface in the cell
        offset *= 0.5;

        for (i = 0; i < nv; ++i) {
            (*c)[i] -= offset;
        }

        ablate::levelSet::Utilities::VOF(dm, p, *c, &vof, &area, NULL);
        vofError = targetVOF - vof;
    };
}

// Returns the VOF for a given cell using the level-set values at the cell vertices.
// Refer to "Quadrature rules for triangular and tetrahedral elements with generalized functions"
//  by Holdych, Noble, and Secor, Int. J. Numer. Meth. Engng 2008; 73:1310-1327.
void ablate::levelSet::Utilities::VOF(DM dm, const PetscInt p, PetscReal *c, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    DMPolytopeType ct;
    PetscInt Nc;
    PetscReal *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Get the cell type and call appropriate VOF function
    DMPlexGetCellType(dm, p, &ct) >> ablate::utilities::PetscUtilities::checkError;
    switch (ct) {
        case DM_POLYTOPE_SEGMENT:
            VOF_1D(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_TRIANGLE:
            VOF_2D_Tri(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_QUADRILATERAL:
            VOF_2D_Quad(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_TETRAHEDRON:
            VOF_3D_Tetra(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_HEXAHEDRON:
            VOF_3D_Hex(coords, c, vof, area, vol);
            break;
        default:
            throw std::invalid_argument("No element geometry for cell " + std::to_string(p) + " with type " + DMPolytopeTypes[ct]);
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;
}

// Returns the VOF for a given cell with a known level set value (c0) and normal (nIn).
//  This computes the level-set values at the vertices by approximating the interface as a straight-line with the same normal
//  as provided
void ablate::levelSet::Utilities::VOF(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *nIn, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    PetscReal *c = NULL;
    ablate::levelSet::Utilities::VertexLevelSet_LS(dm, p, c0, nIn, &c);

    ablate::levelSet::Utilities::VOF(dm, p, c, vof, area, vol);  // Do the actual calculation.

    PetscFree(c) >> ablate::utilities::PetscUtilities::checkError;
}

// Returns the VOF for a given cell using an analytic level set equation
// Refer to "Quadrature rules for triangular and tetrahedral elements with generalized functions"
void ablate::levelSet::Utilities::VOF(DM dm, PetscInt p, const std::shared_ptr<ablate::mathFunctions::MathFunction> &phi, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    PetscInt dim, Nc, nVerts, i;
    PetscReal *c = NULL, *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;

    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Number of vertices
    nVerts = Nc / dim;

    PetscMalloc1(nVerts, &c) >> ablate::utilities::PetscUtilities::checkError;

    // The level set value of each vertex. This assumes that the interface is a line/plane
    //    with the given unit normal.
    for (i = 0; i < nVerts; ++i) {
        c[i] = phi->Eval(&coords[i * dim], dim, 0.0);
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    ablate::levelSet::Utilities::VOF(dm, p, c, vof, area, vol);  // Do the actual calculation.

    PetscFree(c) >> ablate::utilities::PetscUtilities::checkError;
}

// Return the VOF in a cell where the level set is defined at vertices
void ablate::levelSet::Utilities::VOF(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt cell, const ablate::domain::Field *lsField, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    DM dm = subDomain->GetFieldDM(*lsField);
    Vec vec = subDomain->GetVec(*lsField);
    const PetscScalar *array;
    PetscReal *c;

    PetscInt nv, *verts;
    DMPlexCellGetVertices(dm, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    DMGetWorkArray(dm, nv, MPI_REAL, &c);

    VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt i = 0; i < nv; ++i) {
        const PetscReal *val = nullptr;
        xDMPlexPointLocalRead(dm, verts[i], lsField->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;
        c[i] = *val;
    }
    VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

    ablate::levelSet::Utilities::VOF(dm, cell, c, vof, area, vol);

    DMRestoreWorkArray(dm, nv, MPI_REAL, &c);
    DMPlexCellRestoreVertices(dm, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
}



void SaveVertexData(DM dm, Vec vec, const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

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

void SaveVertexData(DM dm, Vec vec, const char fname[255], const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {
  SaveVertexData(dm, vec, fname, field, 1, subDomain);
}

void SaveVertexData(const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  Vec           vec = subDomain->GetVec(*field);
  DM            dm  = subDomain->GetFieldDM(*field);
  SaveVertexData(dm, vec, fname, field, Nc, subDomain);
}

void SaveCellData(DM dm, const Vec vec, const char fname[255], const PetscInt id, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

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

void SaveCellData(DM dm, const Vec vec, const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {
  SaveCellData(dm, vec, fname, field->id, Nc, subDomain);
}


// Inter-processor ghost cells are iterated over, so everything should work fine
static void CutCellLevelSetValues(std::shared_ptr<ablate::domain::SubDomain> subDomain, ablate::domain::Range cellRange, ablate::domain::Range vertRange, ablate::domain::ReverseRange reverseVertRange, const PetscInt *cellMask, DM vofDM, Vec vofVec, const PetscInt vofID, DM lsDM, Vec lsVec, const PetscInt normalID, const PetscInt lsID) {

  const PetscScalar *vofArray = nullptr;
  PetscScalar *lsArray = nullptr;
  PetscInt *lsCount;


  VecGetArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(lsVec, &lsArray) >> ablate::utilities::PetscUtilities::checkError;

  DMGetWorkArray(lsDM, vertRange.end - vertRange.start, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(lsCount, vertRange.end - vertRange.start) >> ablate::utilities::PetscUtilities::checkError;
  lsCount -= vertRange.start;

  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    PetscInt vert = vertRange.GetPoint(v);
    PetscReal *lsVal = nullptr;
    xDMPlexPointLocalRef(lsDM, vert, lsID, lsArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
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
      xDMPlexPointLocalRead(vofDM, cell, vofID, vofArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      // The pre-computed cell-centered normal
      const PetscScalar *n = nullptr;
      xDMPlexPointLocalRead(lsDM, cell, normalID, lsArray, &n) >> ablate::utilities::PetscUtilities::checkError;

      PetscInt nv, *verts;
      DMPlexCellGetVertices(vofDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal *lsVertVals = NULL;
      DMGetWorkArray(lsDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;

      // Level set values at the vertices
      ablate::levelSet::Utilities::VertexLevelSet_VOF(lsDM, cell, *vofVal, n, &lsVertVals);

      for (PetscInt v = 0; v < nv; ++v) {
        PetscScalar *lsVal = nullptr;
        xDMPlexPointLocalRef(lsDM, verts[v], lsID, lsArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
        *lsVal += lsVertVals[v];

        PetscInt vert_i = reverseVertRange.GetIndex(verts[v]);
        ++lsCount[vert_i];
      }

      DMRestoreWorkArray(lsDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexCellRestoreVertices(vofDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

    }
  }

  // This is no longer needed
  VecRestoreArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    if ( lsCount[v] > 0 ) {

      PetscInt vert = vertRange.GetPoint(v);

      PetscReal *lsVal = nullptr;
      xDMPlexPointLocalRef(lsDM, vert, lsID, lsArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

      *lsVal /= lsCount[v];
    }
  }

  lsCount += vertRange.start;
  DMRestoreWorkArray(lsDM, vertRange.end - vertRange.start, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;

  VecRestoreArray(lsVec, &lsArray) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->UpdateAuxLocalVector();

}




//struct reinitCTX {
//    std::shared_ptr<ablate::domain::rbf::RBF> rbf;
//};
//static Parameters parameters{};
#include "domain/RBF/ga.hpp"
#include "domain/RBF/hybrid.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/phs.hpp"
#include "domain/RBF/rbf.hpp"
static std::shared_ptr<ablate::domain::rbf::RBF> vertRBF = nullptr;

static std::shared_ptr<ablate::domain::rbf::RBF> cellRBF = nullptr;

// Temporary for the review
//static PetscInt **cellNeighbors = nullptr, *numberNeighbors = nullptr;

// Make sure that the work is being done on valid cells and not ghost cells
bool ablate::levelSet::Utilities::ValidCell(DM dm, PetscInt p) {
    DMPolytopeType ct;
    DMPlexGetCellType(dm, p, &ct) >> ablate::utilities::PetscUtilities::checkError;

    return (ct < 12);
}


PetscReal GaussianDerivativeFactor(const PetscInt dim, const PetscReal *x, const PetscReal s,  const PetscInt dx, const PetscInt dy, const PetscInt dz) {

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


static PetscInt FindCell(DM dm, const PetscReal x0[], const PetscInt nCells, const PetscInt cells[], PetscReal *distOut) {
  // Return the cell with the cell-center that is the closest to a given point

  PetscReal dist = PETSC_MAX_REAL;
  PetscInt closestCell = -1;
  PetscInt dim;
  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = 0; c < nCells; ++c) {
    PetscReal x[dim];
    DMPlexComputeCellGeometryFVM(dm, cells[c], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

    ablate::utilities::MathUtilities::Subtract(dim, x, x0, x);
    PetscReal cellDist = ablate::utilities::MathUtilities::MagVector(dim, x);
    if (cellDist < dist) {
      closestCell = cells[c];
      dist = cellDist;
    }

  }

  if (distOut) *distOut = dist;

  return (closestCell);


}

static PetscInt *interpCellList = nullptr;


//   Hermite-Gauss quadrature points
static PetscInt nQuad = 4; // Size of the 1D quadrature

//   The quadrature is actually sqrt(2) times the quadrature points. This is as we are integrating
//      against the normal distribution, not exp(-x^2)
static PetscReal quad[4] = {-0.74196378430272585764851359672636022482952014750891895361147387899499975465000530,
                           0.74196378430272585764851359672636022482952014750891895361147387899499975465000530,
                          -2.3344142183389772393175122672103621944890707102161406718291603341725665622712306,
                           2.3344142183389772393175122672103621944890707102161406718291603341725665622712306};

// The weights are the true weights divided by sqrt(pi)
static PetscReal weights[4] = {0.45412414523193150818310700622549094933049562338805584403605771393758003145477625,
                             0.45412414523193150818310700622549094933049562338805584403605771393758003145477625,
                             0.045875854768068491816892993774509050669504376611944155963942286062419968545223748,
                             0.045875854768068491816892993774509050669504376611944155963942286062419968545223748};

static PetscReal sigmaFactor = 1.0;

void BuildInterpCellList(DM dm, const ablate::domain::Range cellRange) {


  PetscReal h;
  DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size
  const PetscReal sigma = sigmaFactor*h;


  PetscInt dim;
  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;


  PetscMalloc1(16*(cellRange.end - cellRange.start), &interpCellList);

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    PetscReal x0[dim];
    DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    PetscInt nCells, *cellList;
    DMPlexGetNeighbors(dm, cell, 2, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt i = 0; i < nQuad; ++i) {
      for (PetscInt j = 0; j < nQuad; ++j) {

        PetscReal x[2] = {x0[0] + sigma*quad[i], x0[1] + sigma*quad[j]};

        const PetscInt interpCell = FindCell(dm, x, nCells, cellList, NULL);

        if (interpCell < 0) {
          PetscInt dim;
          DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;
          int rank;
          MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
          throw std::runtime_error("BuildInterpCellList could not determine the location of (" + std::to_string(x[0]) + ", " + std::to_string(x[1]) + ") on rank " + std::to_string(rank) + ".");
        }

        interpCellList[(c - cellRange.start)*16 + i*4 + j] = interpCell;
      }
    }

    DMPlexRestoreNeighbors(dm, cell, 2, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

  }
}



// Calculate the curvature from a vertex-based level set field using Gaussian convolution.
// Right now this is just 2D for testing purposes.
void CurvatureViaGaussian(DM dm, const PetscInt c, const PetscInt cell, const Vec vec, const ablate::domain::Field *lsField, double *H) {

  PetscInt dim;
  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  PetscReal h;
  DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

//  const PetscInt nQuad = 3; // Size of the 1D quadrature
//  const PetscReal quad[] = {0.0, PetscSqrtReal(3.0), -PetscSqrtReal(3.0)};
//  const PetscReal weights[] = {2.0/3.0, 1.0/6.0, 1.0/6.0};

  PetscReal x0[dim], vol;
  DMPlexComputeCellGeometryFVM(dm, cell, &vol, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  // Mabye relate this to PETSC_SQRT_MACHINE_EPSILON or similar?
  //  The would probably require that the derivative factor be re-done to account for round-off.
  const PetscReal sigma = sigmaFactor*h;

  PetscReal cx = 0.0, cy = 0.0, cxx = 0.0, cyy = 0.0, cxy = 0.0;

  for (PetscInt i = 0; i < nQuad; ++i) {
    for (PetscInt j = 0; j < nQuad; ++j) {

      const PetscReal dist[2] = {sigma*quad[i], sigma*quad[j]};
      PetscReal x[2] = {x0[0] + dist[0], x0[1] + dist[1]};

//      const PetscInt interpCell = FindCell(dm, x, nCells, cellList, NULL);
      const PetscInt interpCell = interpCellList[c*16 + i*4 + j];

      const PetscReal lsVal = vertRBF->Interpolate(lsField, vec, interpCell, x);

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


// Calculate the curvature from a vertex-based level set field using Gaussian convolution.
// Right now this is just 2D for testing purposes.
PetscReal LaplacianViaGaussian(DM dm, const PetscInt c, const PetscInt cell, const Vec vec, const ablate::domain::Field *lsField) {

  PetscInt dim;
  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  PetscReal h;
  DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

//  const PetscInt nQuad = 3; // Size of the 1D quadrature
//  const PetscReal quad[] = {0.0, PetscSqrtReal(3.0), -PetscSqrtReal(3.0)};
//  const PetscReal weights[] = {2.0/3.0, 1.0/6.0, 1.0/6.0};

  PetscReal x0[dim], vol;
  DMPlexComputeCellGeometryFVM(dm, cell, &vol, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  // Mabye relate this to PETSC_SQRT_MACHINE_EPSILON or similar?
  //  The would probably require that the derivative factor be re-done to account for round-off.
  const PetscReal sigma = sigmaFactor*h;

  PetscReal cxx = 0.0, cyy = 0.0;

  for (PetscInt i = 0; i < nQuad; ++i) {
    for (PetscInt j = 0; j < nQuad; ++j) {

      const PetscReal dist[2] = {sigma*quad[i], sigma*quad[j]};
      PetscReal x[2] = {x0[0] + dist[0], x0[1] + dist[1]};

//      const PetscInt interpCell = FindCell(dm, x, nCells, cellList, NULL);
      const PetscInt interpCell = interpCellList[c*16 + i*4 + j];

      const PetscReal lsVal = vertRBF->Interpolate(lsField, vec, interpCell, x);

      const PetscReal wt = weights[i]*weights[j];

      cxx += wt*GaussianDerivativeFactor(dim, dist, sigma, 2, 0, 0)*lsVal;
      cyy += wt*GaussianDerivativeFactor(dim, dist, sigma, 0, 2, 0)*lsVal;
    }
  }

  return (cxx + cyy);

}



// Calculate the curvature from a vertex-based level set field using Gaussian convolution.
// Right now this is just 2D for testing purposes.
PetscReal SmoothingViaGaussian(DM dm, const PetscInt c, const PetscInt cell, const Vec vec, const ablate::domain::Field *field) {

  PetscInt dim;
  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  PetscReal h;
  DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

//  const PetscInt nQuad = 3; // Size of the 1D quadrature
//  const PetscReal quad[] = {0.0, PetscSqrtReal(3.0), -PetscSqrtReal(3.0)};
//  const PetscReal weights[] = {2.0/3.0, 1.0/6.0, 1.0/6.0};

  PetscReal x0[dim], vol;
  DMPlexComputeCellGeometryFVM(dm, cell, &vol, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  // Mabye relate this to PETSC_SQRT_MACHINE_EPSILON or similar?
  //  The would probably require that the derivative factor be re-done to account for round-off.
  const PetscReal sigma = sigmaFactor*h;

  PetscReal newVal = 0.0;

  for (PetscInt i = 0; i < nQuad; ++i) {
    for (PetscInt j = 0; j < nQuad; ++j) {

      const PetscReal dist[2] = {sigma*quad[i], sigma*quad[j]};
      PetscReal x[2] = {x0[0] + dist[0], x0[1] + dist[1]};

      const PetscInt interpCell = interpCellList[c*16 + i*4 + j];

      const PetscReal lsVal = vertRBF->Interpolate(field, vec, interpCell, x);

      const PetscReal wt = weights[i]*weights[j];

      newVal += wt*GaussianDerivativeFactor(dim, dist, sigma, 0, 0, 0)*lsVal;
    }
  }

  return (newVal);

}




//#define saveData

#ifdef saveData
static PetscInt saveIter = 0;
#endif




//vofField: cell-based field containing the target volume-of-fluid
//lsField: vertex-based field for level set values
//normalField: cell-based vector field containing normals
//curvField: cell-based vector field containing curvature
void ablate::levelSet::Utilities::Reinitialize(
  const ablate::finiteVolume::FiniteVolumeSolver &flow,
  std::shared_ptr<ablate::domain::SubDomain> subDomain,
  const Vec solVec,
  const ablate::domain::Field *vofField,
  const PetscInt nLevels,
  const ablate::domain::Field *lsField,
  const ablate::domain::Field *vertexNormalField,
  const ablate::domain::Field *cellNormalField,
  const ablate::domain::Field *curvField
) {


#ifdef saveData
  ++saveIter;
#endif

  // Note: Need to write a unit test where the vof and ls fields aren't in the same DM, e.g. one is a SOL vector and one is an AUX vector.

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
//printf("%+f\n", h);

  /***********************************************************************************************/
  // THIS IS TEMPORARY AND NEEDS TO BE MOVED TO THE YAML FILE OR SOMETHING ELSE AFTER THE REVIEW
  /***********************************************************************************************/
  if ( vertRBF==nullptr ) {
    // nLevels needs to be wide enough to support the width necessary for the RBF.
    PetscInt polyAug = 2; // Looks like I need an odd augmented polynomial order for the curvature to be acceptable
    bool doesNotHaveDerivatives = false;
    bool doesNotHaveInterpolation = false;
    bool returnNeighborVertices = true;
    vertRBF = std::make_shared<ablate::domain::rbf::IMQ>(polyAug, 1e-2*h, doesNotHaveDerivatives, doesNotHaveInterpolation, returnNeighborVertices);

    vertRBF->Setup(subDomain);       // This causes issues (I think)
    vertRBF->Initialize();  //         Initialize
  }

  if ( cellRBF==nullptr ) {
    PetscInt polyAug = 2;
    bool doesNotHaveDerivatives = false;
    bool doesNotHaveInterpolation = false;
    bool returnNeighborVertices = false;
    cellRBF = std::make_shared<ablate::domain::rbf::MQ>(polyAug, h, doesNotHaveDerivatives, doesNotHaveInterpolation, returnNeighborVertices);

    cellRBF->Setup(subDomain);       // This causes issues (I think)
    cellRBF->Initialize();  //         Initialize
  }


  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;


  ablate::domain::Range cellRange, vertRange;
  subDomain->GetCellRange(nullptr, cellRange);
  subDomain->GetRange(nullptr, 0, vertRange);

  ablate::domain::Range cellRangeWithoutGhost;
  flow.GetCellRangeWithoutGhost(cellRangeWithoutGhost);

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


  Vec workVec, workVecGlobal;
  PetscScalar *workArray = nullptr;
  DMGetLocalVector(auxDM, &workVec);
  DMGetGlobalVector(auxDM, &workVecGlobal);
  VecGetArray(workVec, &workArray);
  VecZeroEntries(workVec);

#ifdef saveData
  char fname[255];
  sprintf(fname, "vof0_%03ld.txt", saveIter);
  SaveCellData(solDM, solVec, fname, vofField, 1, subDomain);
#endif

  if ( interpCellList==nullptr ) {
    BuildInterpCellList(auxDM, cellRangeWithoutGhost);
  }

for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
  const PetscInt vert = vertRange.GetPoint(v);

  PetscScalar *smoothVOF = nullptr;
  xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &smoothVOF);
  *smoothVOF = 0.0;

  PetscInt nCells, *cellList;
  DMPlexVertexGetCells(auxDM, vert, &nCells, &cellList);
  for (PetscInt i = 0; i < nCells; ++i) {
    const PetscScalar *vof = nullptr;
    xDMPlexPointLocalRead(solDM, cellList[i], vofID, solArray, &vof);

    *smoothVOF += *vof;
  }
  *smoothVOF /= nCells;

  DMPlexVertexRestoreCells(auxDM, vert, &nCells, &cellList);
}
subDomain->UpdateAuxLocalVector();

for (PetscInt c = cellRangeWithoutGhost.start; c < cellRangeWithoutGhost.end; ++c){
  const PetscInt cell = cellRangeWithoutGhost.GetPoint(c);

  PetscScalar *cellVOF = nullptr;
  xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &cellVOF);
  *cellVOF = 0.0;

  PetscInt nVert, *vertList;
  DMPlexCellGetVertices(auxDM, cell, &nVert, &vertList);
  for (PetscInt i = 0; i < nVert; ++i) {
    PetscScalar *vertVOF = nullptr;
    xDMPlexPointLocalRef(auxDM, vertList[i], lsID, auxArray, &vertVOF);
    *cellVOF += *vertVOF;
  }

  *cellVOF /= nVert;
  DMPlexCellRestoreVertices(auxDM, cell, &nVert, &vertList);
}

VecRestoreArray(workVec, &workArray);
DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;
VecGetArray(workVec, &workArray);


#ifdef saveData
  sprintf(fname, "vof1_%03ld.txt", saveIter);
  SaveCellData(auxDM, workVec, fname, vofField, 1, subDomain);
#endif



/**************** Determine the cut-cells  *************************************/

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    PetscInt cell = cellRange.GetPoint(c);

    if (ablate::levelSet::Utilities::ValidCell(solDM, cell)) {
      const PetscScalar *vofVal = nullptr;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;
      if ( ((*vofVal) > 0.001) && ((*vofVal) < 0.999) ) {
        cellMask[c] = 1;    // Mark as a cut-cell
      }
    }
  }

#ifdef saveData
{
  sprintf(fname, "mask0_%03ld.txt", saveIter);
  FILE *f1 = fopen(fname, "w");
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    const PetscInt cell = cellRange.GetPoint(c);
    PetscReal x[dim];
    DMPlexComputeCellGeometryFVM(solDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%+f\t", x[d]);
    fprintf(f1, "%ld\n", cellMask[c]);
  }
  fclose(f1);
}
#endif


// Turn off any "cut cells" where the cell is not surrounded by any other cut cells.
// To avoid cut-cells two cells-thick turn off any cut-cells which have a neighoring gradient passing through them.
for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

  if (cellMask[c]==1) {

    const PetscInt cell = cellRange.GetPoint(c);

    PetscInt nCells, *cells;
    DMPlexGetNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

    PetscInt nCut = 0;
    for (PetscInt i = 0; i < nCells; ++i) {
      PetscInt id = reverseCellRange.GetIndex(cells[i]);
      nCut += (cellMask[id] > 0);
    }

    DMPlexRestoreNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

    cellMask[c] = (nCut==1 ? 2 : 1); // If nCut equals 1 then the center cell is the only cut cell, so temporarily mark it as 2.


    const PetscScalar *n = nullptr;
    xDMPlexPointLocalRead(auxDM, cell, cellNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError; // VOF normal

    if (cellMask[c]==1 && ablate::utilities::MathUtilities::MagVector(dim, n)>PETSC_SMALL) {
      // Now check for two-deep cut-cells.
      const PetscScalar *n = nullptr;
      xDMPlexPointLocalRead(auxDM, cell, cellNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError; // VOF normal


      const PetscScalar *centerVOF = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &centerVOF) >> ablate::utilities::PetscUtilities::checkError;

      const PetscReal direction[2] = {-1.0, +1.0};
      for (PetscInt d = 0; d < 2; ++d) {
        PetscInt neighborCell = -1;
        DMPlexGetForwardCell(auxDM, cell, n, direction[d], &neighborCell) >> ablate::utilities::PetscUtilities::checkError;
        if (neighborCell > -1) {

          const PetscReal *neighborVOF;
          xDMPlexPointLocalRef(auxDM, neighborCell, vofID, workArray, &neighborVOF) >> ablate::utilities::PetscUtilities::checkError;

          if (PetscAbsReal(*neighborVOF - 0.5) < PetscAbsReal(*centerVOF - 0.5)) {
            cellMask[c] = 2;
            break;
          }
        }
      }
    }
  }
}

for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
  cellMask[c] = (cellMask[c]==2 ? 0 : cellMask[c]);
}

/**************** Determine the initial unit normal *************************************/


  DM cutCellDM = solDM;
  Vec cutCellVec = solVec;

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    PetscInt cell = cellRange.GetPoint(c);

    PetscScalar *n = nullptr;
    xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
    for (PetscInt d = 0; d < dim; ++d ) n[d] = 0.0;

    if (cellMask[c]==1) {

      // Will this crap near the edges of a processor?
      DMPlexCellGradFromCell(cutCellDM, cell, cutCellVec, vofID, 0, n);
//      if ( dim > 0 ) n[0] = cellRBF->EvalDer(auxDM, workVec, vofID, cell, 1, 0, 0);
//      if ( dim > 1 ) n[1] = cellRBF->EvalDer(auxDM, workVec, vofID, cell, 0, 1, 0);
//      if ( dim > 2 ) n[2] = cellRBF->EvalDer(auxDM, workVec, vofID, cell, 0, 0, 1);

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


  subDomain->UpdateAuxLocalVector();



#ifdef saveData
{
  sprintf(fname, "mask1_%03ld.txt", saveIter);
  FILE *f1 = fopen(fname, "w");
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    const PetscInt cell = cellRange.GetPoint(c);
    PetscReal x[dim];
    DMPlexComputeCellGeometryFVM(solDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%+f\t", x[d]);
    fprintf(f1, "%ld\n", cellMask[c]);
  }
  fclose(f1);
  sprintf(fname, "cellNormal0_%03ld.txt", saveIter);
  SaveCellData(auxDM, auxVec, fname, cellNormalField, dim, subDomain);
}
#endif







/**************** Iterate to get the level-set values at vertices *************************************/

  // Temporary level-set work array to store old values
  PetscScalar *tempLS;
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
  tempLS -= vertRange.start;

  PetscReal maxDiff = 1.0;
  PetscInt iter = 0;

  MPI_Comm auxCOMM = PetscObjectComm((PetscObject)auxDM);

//SaveCellData(auxDM, auxVec, "normal0.txt", cellNormalField, dim, subDomain);


  while ( maxDiff > 1e-3*h && iter<100 ) {

    ++iter;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v]==1) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *oldLS = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &oldLS) >> ablate::utilities::PetscUtilities::checkError;
        tempLS[v] = *oldLS;
      }
    }

    // Note: The unit normal and CutCellLevelSetValues must work on the same set of datat.

    // This updates the lsField by taking the average vertex values necessary to match the VOF in cutcells
    CutCellLevelSetValues(subDomain, cellRange, vertRange, reverseVertRange, cellMask, cutCellDM, cutCellVec, vofID, auxDM, auxVec, cellNormalID, lsID);

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
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      if (vertMask[v] == 1) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *newLS = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &newLS) >> ablate::utilities::PetscUtilities::checkError;

        maxDiff = PetscMax(maxDiff, PetscAbsReal(tempLS[v] - *newLS));
      }
    }
    // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);
#ifdef saveData
    PetscPrintf(PETSC_COMM_WORLD, "Cut Cells %" PetscInt_FMT": %+e\n", iter, maxDiff) >> ablate::utilities::PetscUtilities::checkError;
#endif
  }

  if (maxDiff > 1e-3*h) {
    throw std::runtime_error("Interface reconstruction has failed.\n");
  }


#ifdef saveData
  sprintf(fname, "ls1_%03ld.txt", saveIter);
  SaveVertexData(auxDM, auxVec, fname, lsField, 1, subDomain);
#endif


/**************** Set the data in the rest of the domain to be a large value *************************************/
//PetscPrintf(PETSC_COMM_WORLD, "Setting data\n");
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



/**************** Mark the cells that need to be udpated via the reinitialization equation *************************************/
//PetscPrintf(PETSC_COMM_WORLD, "Marking cells\n");
  // Mark all of the cells neighboring cells level-by-level.
  // Note that DMPlexGetNeighbors has an issue in parallel whereby cells will be missed due to the unknown partitioning -- Need to investigate

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);
    PetscScalar *maskVal = nullptr;
    xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;
    *maskVal = cellMask[c];
  }


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




  for (PetscInt l = 1; l <= nLevels; ++l) {


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

                *lsVal = lsRange[ *vofVal < 0.5 ? 1 : 0 ];

              }
            }

            DMPlexCellRestoreVertices(auxDM, cells[i], &nv, &verts);
          }
        }
        DMPlexRestoreNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      }
    }


    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;
  }

  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->UpdateAuxLocalVector();


#ifdef saveData
{
  sprintf(fname, "mask2_%03ld.txt", saveIter);
  FILE *f1 = fopen(fname, "w");
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    const PetscInt cell = cellRange.GetPoint(c);
    PetscReal x[dim];
    DMPlexComputeCellGeometryFVM(solDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt d = 0; d < dim; ++d) fprintf(f1, "%+f\t", x[d]);
    fprintf(f1, "%ld\n", cellMask[c]);
  }
  fclose(f1);
  sprintf(fname, "ls2_%03ld.txt", saveIter);
  SaveVertexData(auxDM, auxVec, fname, lsField, 1, subDomain);
}
#endif



//exit(0);
//PetscPrintf(PETSC_COMM_WORLD, "Reinit\n");
/**************** Level-set reinitialization equation *************************************/
  const PetscInt vertexNormalID = vertexNormalField->id;
  const PetscInt curvID = curvField->id;



  maxDiff = 1.0;
  iter = 0;

//  while (maxDiff>1.e-2 && iter<(nLevels*5)) {
  while (maxDiff>1.e-2 && iter<3*(nLevels+1)) {
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

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 0) {
        PetscInt vert = vertRange.GetPoint(v);

        PetscReal *g = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, g) >> ablate::utilities::PetscUtilities::checkError;

//        // Check if the gradient is zero. This occurs along the skeleton of a shape
//        PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);
//        if (nrm < PETSC_SMALL) {
//          DMPlexComputeCellGeometryFVM(auxDM, vert, NULL, g, NULL) >> ablate::utilities::PetscUtilities::checkError;
//          for (PetscInt d=0; d<dim; ++d) g[d] += 10.0*PETSC_SMALL;
//        }

      }
    }

    subDomain->UpdateAuxLocalVector();


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

        PetscReal sgn = (*phi)/PetscSqrtReal(PetscSqr(*phi) + PetscSqr(h));

        if (ablate::utilities::MathUtilities::MagVector(dim, g) < 1.e-10) {
          *phi = tempLS[v] + 0.5*h*sgn;
        }
        else {

          VertexUpwindGrad(auxDM, auxArray, cellNormalID, vert, PetscSignReal(*phi), g);

          PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);

          *phi = tempLS[v] - 0.5*h*sgn*(nrm - 1.0);

          // In parallel runs VertexUpwindGrad may return g=0 as there aren't any upwind nodes. Don't incldue that in the diff check
          if (ablate::utilities::MathUtilities::MagVector(dim, g) > PETSC_SMALL) maxDiff = PetscMax(maxDiff, PetscAbsReal(nrm - 1.0));
        }


      }
    }

    subDomain->UpdateAuxLocalVector();


     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);


#ifdef saveData
    PetscPrintf(PETSC_COMM_WORLD, "Reinit %3" PetscInt_FMT": %e\n", iter, maxDiff);
#endif

  }

#ifdef saveData
  sprintf(fname, "ls3_%03ld.txt", saveIter);
  SaveVertexData(auxDM, auxVec, fname, lsField, 1, subDomain);
#endif

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

#ifdef saveData
  sprintf(fname, "mask3_%03ld.txt", saveIter);
  SaveCellData(auxDM, workVec, fname, vofField, 1, subDomain);
#endif

  for (PetscInt c = cellRangeWithoutGhost.start; c < cellRangeWithoutGhost.end; ++c) {
    PetscInt cell = cellRangeWithoutGhost.GetPoint(c);
    PetscScalar *H = nullptr;
    xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &H);

    PetscScalar *maskVal;
    xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;

//    if ((PetscAbsScalar(*maskVal - 1.0) < PETSC_SMALL) && ablate::levelSet::Utilities::ValidCell(auxDM, cell)) {
    if ( (*maskVal > 0.5) && (*maskVal < (nLevels-1)) && ablate::levelSet::Utilities::ValidCell(auxDM, cell)) {
      CurvatureViaGaussian(auxDM, c - cellRangeWithoutGhost.start, cell, auxVec, lsField, H);
    }
    else {
      *H = 0.0;
    }
  }

  subDomain->UpdateAuxLocalVector();
#ifdef saveData
  sprintf(fname, "curv0_%03ld.txt", saveIter);
  SaveCellData(auxDM, auxVec, fname, curvID, 1, subDomain);
#endif


  // Extension
  PetscInt vertexCurvID = lsID; // Store the vertex curvatures in the work vec at the same location as the level-set


  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    if (vertMask[v] > 0 && vertMask[v] < nLevels - 1) {
      PetscInt vert = vertRange.GetPoint(v);
      PetscReal *H = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &H) >> ablate::utilities::PetscUtilities::checkError;

      *H = 0.0;

      PetscInt nCells, *cells, nAve = 0;
      DMPlexVertexGetCells(auxDM, vert, &nCells, &cells);

      for (PetscInt c = 0; c < nCells; ++c) {

        const PetscInt cm = cellMask[reverseCellRange.GetIndex(cells[c])];

        if (cm > 0 ) {

          PetscScalar *cellH = nullptr;
          xDMPlexPointLocalRef(auxDM, cells[c], curvID, auxArray, &cellH);
          *H += *cellH;
          ++nAve;
        }
      }


      *H /= nAve;

      DMPlexVertexRestoreCells(auxDM, vert, &nCells, &cells);

    }
  }


  DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
  DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;



#ifdef saveData
  sprintf(fname, "vertH0_%03ld.txt", saveIter);
  SaveVertexData(auxDM, workVec, fname, lsField, 1, subDomain);
#endif


  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    if (vertMask[v] > 0) {
      PetscInt vert = vertRange.GetPoint(v);

      PetscReal *n = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
      ablate::utilities::MathUtilities::NormVector(dim, n, n);
    }
  }
  subDomain->UpdateAuxLocalVector();

  maxDiff = PETSC_MAX_REAL;
  iter = 0;
  while ( maxDiff>5e-2 && iter<3*(nLevels+1)) {
    ++iter;

    // Curvature gradient at the cell-center
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] > 0) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscScalar *g = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, workArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, workVec, vertexCurvID, 0, g);
      }
    }

    maxDiff = -PETSC_MAX_REAL;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 1) {
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

        PetscReal *mag = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, workArray, &mag) >> ablate::utilities::PetscUtilities::checkError;
        mag[0] = PetscAbsReal(dH);
      }
    }

    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;

//     This is temporary until after the review.
//     The norm magnitude is incorrect at the edge of processor domains. There needs to be a way to identify
//      cell which are ghost cells as they will have incorrect answers.

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 1) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *mag = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, vertexNormalID, workArray, &mag) >> ablate::utilities::PetscUtilities::checkError;
        maxDiff = PetscMax(maxDiff, PetscAbsReal(mag[0]));
      }
    }

     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

#ifdef saveData
    PetscPrintf(PETSC_COMM_WORLD, "Extension %3" PetscInt_FMT": %e\n", iter, maxDiff);
#endif
  }


#ifdef saveData
  sprintf(fname, "vertH1_%03ld.txt", saveIter);
  SaveVertexData(auxDM, workVec, fname, lsField, 1, subDomain);
#endif



   for (PetscInt iter = 0; iter < 5; ++iter) {

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] > 0) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscScalar *g = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, workArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, workVec, vertexCurvID, 0, g);

        const PetscScalar *n = nullptr;
        xDMPlexPointLocalRead(auxDM, cell, cellNormalID, auxArray, &n);

        const PetscReal dot = ablate::utilities::MathUtilities::DotVector(dim, n, g);

        for (PetscInt d = 0; d < dim; ++d) g[d] -= dot*n[d];

      }
    }
    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;


    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 0) {
        PetscInt vert = vertRange.GetPoint(v);
        PetscReal div = 0.0;

        for (PetscInt d = 0; d < dim; ++d) {
          PetscReal g[dim];
          DMPlexVertexGradFromCell(auxDM, vert, workVec, cellNormalID, d, g);
          div += g[d];
        }

        PetscReal *H = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &H);

        *H += 0.5*h*h*div;

      }
    }
    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;
  }



#ifdef saveData
  sprintf(fname, "vertH2_%03ld.txt", saveIter);
  SaveVertexData(auxDM, workVec, fname, lsField, 1, subDomain);
#endif



  // Now set the curvature at the cell-center via averaging

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    if (cellMask[c] > 0) {
      PetscInt cell = cellRange.GetPoint(c);

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

      DMPlexCellRestoreVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }


  subDomain->UpdateAuxLocalVector();

#ifdef saveData
  sprintf(fname, "cellH0_%03ld.txt", saveIter);
  SaveVertexData(auxDM, workVec, fname, lsField, 1, subDomain);
  sprintf(fname, "cellNormal1_%03ld.txt", saveIter);
  SaveCellData(auxDM, auxVec, fname, cellNormalField, dim, subDomain);

#endif

  VecRestoreArray(workVec, &workArray);
  DMRestoreLocalVector(auxDM, &workVec) >> utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(auxDM, &workVecGlobal) >> utilities::PetscUtilities::checkError;



  // Cleanup all memory
  tempLS += vertRange.start;
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
  vertMask += vertRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  cellMask += cellRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
  DMRestoreWorkArray(solDM, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->RestoreRange(vertRange);
  subDomain->RestoreRange(cellRange);
  flow.RestoreRange(cellRangeWithoutGhost);

  VecRestoreArray(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;

}
