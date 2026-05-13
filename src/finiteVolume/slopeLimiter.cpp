#include "slopeLimiter.hpp"
#include <petsc/private/dmpleximpl.h>
#include <utilities/petscUtilities.hpp>
#include <cmath>

namespace ablate::finiteVolume {

void SlopeLimiter::Setup(DM dm, const domain::Range& cellRange) {
    if (isSetup) return;

    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

    is1D = (dim == 1);

    DMPlexGetMinRadius(dm, &minCellRadius) >> utilities::PetscUtilities::checkError;

    cellToFaces.assign(cEnd - cStart, std::vector<PetscInt>());
    for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscInt* faces;
        PetscInt numFaces;
        DMPlexGetCone(dm, c, &faces) >> utilities::PetscUtilities::checkError;
        DMPlexGetConeSize(dm, c, &numFaces) >> utilities::PetscUtilities::checkError;
        if (numFaces > 0) {
            cellToFaces[c - cStart].assign(faces, faces + numFaces);
        }
    }

    (void)cellRange;  // not used; iteration is driven by ApplyLimiter
    isSetup = true;
}

void SlopeLimiter::ApplyLimiter(DM dm, DM dmGrad, PetscInt dim, const domain::Field& field,
                                const domain::Range& cellRange, Vec cellGeomVec, Vec faceGeomVec,
                                const PetscScalar* xLocalArray, PetscScalar* gradGlobArray) {
    if (!isSetup) {
        throw std::runtime_error("SlopeLimiter must be set up before applying limiter");
    }

    DMLabel ghostLabel = nullptr;
    DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    if (!IsActiveFor(field.name)) {
        const PetscInt nCompZero = field.numberComponents;
        for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
            const PetscInt cell = cellRange.points ? cellRange.points[c] : c;
            if (cell < cStart || cell >= cEnd) continue;
            if (ghostLabel) {
                PetscInt g = -1;
                DMLabelGetValue(ghostLabel, cell, &g) >> utilities::PetscUtilities::checkError;
                if (g >= 0) continue;
            }
            PetscScalar* gradCell = nullptr;
            DMPlexPointGlobalRef(dmGrad, cell, gradGlobArray, &gradCell) >> utilities::PetscUtilities::checkError;
            if (!gradCell) continue;
            for (PetscInt comp = 0; comp < nCompZero; ++comp) {
                for (PetscInt d = 0; d < dim; ++d) {
                    gradCell[comp * dim + d] = 0.0;
                }
            }
        }
        return;
    }

    const PetscScalar* cellGeomArray = nullptr;
    const PetscScalar* faceGeomArray = nullptr;
    DM dmCellGeom = nullptr;
    DM dmFaceGeom = nullptr;
    if (!is1D) {
        VecGetDM(cellGeomVec, &dmCellGeom) >> utilities::PetscUtilities::checkError;
        VecGetDM(faceGeomVec, &dmFaceGeom) >> utilities::PetscUtilities::checkError;
        VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
        VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
    }

    const PetscInt nComp = field.numberComponents;
    const PetscInt fieldId = field.id;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        if (cell < cStart || cell >= cEnd) continue;

        if (ghostLabel) {
            PetscInt g = -1;
            DMLabelGetValue(ghostLabel, cell, &g) >> utilities::PetscUtilities::checkError;
            if (g >= 0) continue;
        }

        PetscScalar* gradCell = nullptr;
        DMPlexPointGlobalRef(dmGrad, cell, gradGlobArray, &gradCell) >> utilities::PetscUtilities::checkError;
        if (!gradCell) continue;

        const PetscScalar* xCell = nullptr;
        DMPlexPointLocalFieldRead(dm, cell, fieldId, xLocalArray, &xCell) >> utilities::PetscUtilities::checkError;
        if (!xCell) continue;

        const PetscFVCellGeom* cg = nullptr;
        if (!is1D) {
            DMPlexPointLocalRead(dmCellGeom, cell, cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;
            if (!cg) continue;
        }

        for (PetscInt comp = 0; comp < nComp; ++comp) {
            const PetscReal cellVal = PetscRealPart(xCell[comp]);

            PetscReal minVal = cellVal;
            PetscReal maxVal = cellVal;

            const auto& faces = cellToFaces[cell - cStart];
            for (PetscInt face : faces) {
                PetscInt supSize = 0;
                const PetscInt* support = nullptr;
                DMPlexGetSupportSize(dm, face, &supSize) >> utilities::PetscUtilities::checkError;
                DMPlexGetSupport(dm, face, &support) >> utilities::PetscUtilities::checkError;
                if (supSize != 2) continue;  // domain-boundary face; no neighbor
                const PetscInt nb = (support[0] == cell) ? support[1] : support[0];

                const PetscScalar* xNb = nullptr;
                DMPlexPointLocalFieldRead(dm, nb, fieldId, xLocalArray, &xNb) >> utilities::PetscUtilities::checkError;
                if (!xNb) continue;
                const PetscReal nbVal = PetscRealPart(xNb[comp]);
                minVal = PetscMin(minVal, nbVal);
                maxVal = PetscMax(maxVal, nbVal);
            }

            PetscReal alpha = 1.0;
            for (PetscInt face : faces) {
                PetscInt supSize = 0;
                const PetscInt* support = nullptr;
                DMPlexGetSupportSize(dm, face, &supSize) >> utilities::PetscUtilities::checkError;
                DMPlexGetSupport(dm, face, &support) >> utilities::PetscUtilities::checkError;
                if (supSize != 2) continue;

                PetscReal r[3] = {0.0, 0.0, 0.0};
                if (is1D) {
                    const auto& fs = cellToFaces[cell - cStart];
                    if (fs.size() == 2) {
                        const PetscInt leftFace = PetscMin(fs[0], fs[1]);
                        r[0] = (face == leftFace) ? -minCellRadius : minCellRadius;
                    } else {
                        r[0] = minCellRadius;
                    }
                } else {
                    const PetscFVFaceGeom* fg = nullptr;
                    DMPlexPointLocalRead(dmFaceGeom, face, faceGeomArray, &fg) >> utilities::PetscUtilities::checkError;
                    if (!fg) continue;
                    for (PetscInt d = 0; d < dim; ++d) {
                        r[d] = fg->centroid[d] - cg->centroid[d];
                    }
                }

                PetscReal delta = 0.0;
                for (PetscInt d = 0; d < dim; ++d) {
                    delta += PetscRealPart(gradCell[comp * dim + d]) * r[d];
                }

                if (PetscAbsReal(delta) < 1e-14) continue;

                PetscReal alpha_f;
                if (delta > 0.0) {
                    alpha_f = (maxVal - cellVal) / delta;
                } else {
                    alpha_f = (minVal - cellVal) / delta;
                }
                if (alpha_f < 0.0) alpha_f = 0.0;
                if (alpha_f > 1.0) alpha_f = 1.0;
                alpha = PetscMin(alpha, alpha_f);
            }

            for (PetscInt d = 0; d < dim; ++d) {
                gradCell[comp * dim + d] *= alpha;
            }
        }
    }

    if (!is1D) {
        VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
        VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
    }
}

}  // namespace ablate::finiteVolume
