#include "domain/RBF/mq.hpp"
#include "intSharp.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include <fstream>
#include <vector>
#include "nPhaseNonconservativeRHS.hpp"

void ablate::finiteVolume::processes::NPhaseNonconservativeRHS::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    NPhaseNonconservativeRHS::subDomain = solver.GetSubDomainPtr();
}
ablate::finiteVolume::processes::NPhaseNonconservativeRHS::NPhaseNonconservativeRHS(const double mInf, const std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> pgs) : pgs(pgs), mInf(mInf) {}
ablate::finiteVolume::processes::NPhaseNonconservativeRHS::~NPhaseNonconservativeRHS() { DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; }

// void nPhaseNonconservativeRHSPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::NPhaseNonconservativeRHS* nPhaseNonconservativeRHSProcess) {
//     nPhaseNonconservativeRHSProcess->PreStage(flowTs, solver, stagetime);
//   }

void ablate::finiteVolume::processes::NPhaseNonconservativeRHS::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    auto dim = flow.GetSubDomain().GetDimensions();
    auto dm = flow.GetSubDomain().GetDM();
    PetscFE fe_coords;
    PetscInt k = 1;



    if ( flow.FindProcess<ablate::finiteVolume::processes::NPhaseAllaireAdvection>() ) {
      throw std::runtime_error("NPhaseNonconservativeRHS process is no longer required for NPhaseAllaireAdvection");
    }



    DMClone(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
    PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
    DMSetField(vertexDM, 0, nullptr, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
    PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;
    DMCreateDS(vertexDM) >> utilities::PetscUtilities::checkError;

    // Get cell and face ranges
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;  // Cells
    DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;  // Faces

    cellToFaces.clear();
    faceToCells.clear();
    cellToFaces.reserve(cEnd - cStart);
    faceToCells.reserve(fEnd - fStart);

    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscInt nFaces;
        const PetscInt* faces;
        DMPlexGetConeSize(dm, cell, &nFaces) >> utilities::PetscUtilities::checkError;
        DMPlexGetCone(dm, cell, &faces) >> utilities::PetscUtilities::checkError;
        cellToFaces[cell].assign(faces, faces + nFaces);
    }

    for (PetscInt face = fStart; face < fEnd; ++face) {
        PetscInt nCells;
        const PetscInt* cells;
        DMPlexGetSupportSize(dm, face, &nCells) >> utilities::PetscUtilities::checkError;
        DMPlexGetSupport(dm, face, &cells) >> utilities::PetscUtilities::checkError;
        faceToCells[face].assign(cells, cells + nCells);
    }

    // Register prestage
    // auto nPhaseNonconservativeRHSPreStage = std::bind(nPhaseNonconservativeRHSPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
    // flow.RegisterPreStage(nPhaseNonconservativeRHSPreStage);

    flow.RegisterRHSFunction(ComputeNonconservativeRHS, this);

    // Register post-step function to enforce bounds
    RegisterPostStep(flow);
}

static inline PetscReal MagVector(PetscInt dim, const PetscReal *in) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    return PetscSqrtReal(mag);
}

void ablate::finiteVolume::processes::NPhaseNonconservativeRHS::ComputeBoundaryDistances() {
    cellBoundaryDistance.clear();
    cellBoundaryDistance.reserve(cEnd - cStart);
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        cellBoundaryDistance[cell] = std::numeric_limits<PetscInt>::max();
    }

    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const auto cellFacesIt = cellToFaces.find(cell);
        if (cellFacesIt == cellToFaces.end()) continue;
        for (PetscInt face : cellFacesIt->second) {
            const auto faceCellsIt = faceToCells.find(face);
            if (faceCellsIt == faceToCells.end()) continue;
            if (faceCellsIt->second.size() == 1) {
                cellBoundaryDistance[cell] = 0;
                break;
            }
        }
    }

    bool changed;
    do {
        changed = false;
        for (PetscInt cell = cStart; cell < cEnd; ++cell) {
            const PetscInt dCur = cellBoundaryDistance[cell];
            if (dCur == std::numeric_limits<PetscInt>::max()) continue;
            const auto cellFacesIt = cellToFaces.find(cell);
            if (cellFacesIt == cellToFaces.end()) continue;
            for (PetscInt face : cellFacesIt->second) {
                const auto faceCellsIt = faceToCells.find(face);
                if (faceCellsIt == faceToCells.end()) continue;
                const auto& cells = faceCellsIt->second;
                if (cells.size() != 2) continue;
                const PetscInt neighbor = (cells[0] == cell) ? cells[1] : cells[0];
                auto neighborIt = cellBoundaryDistance.find(neighbor);
                if (neighborIt == cellBoundaryDistance.end()) continue;
                if (neighborIt->second > dCur + 1) {
                    neighborIt->second = dCur + 1;
                    changed = true;
                }
            }
        }
    } while (changed);
}

PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M1Plus(PetscReal m) { return 0.5 * (m + PetscAbs(m)); }

PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M2Plus(PetscReal m) { return 0.25 * PetscSqr(m + 1); }

PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M1Minus(PetscReal m) { return 0.5 * (m - PetscAbs(m)); }
PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M2Minus(PetscReal m) { return -0.25 * PetscSqr(m - 1); }

PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M4Plus(PetscReal m) {
    if (PetscAbs(m) >= 1.0) {
        return M1Plus(m);
    } else {
        PetscReal beta = 0.125;
        return M2Plus(m) * (1.0 - 16.0 * beta * M2Minus(m));
    }
}
PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::M4Minus(PetscReal m) {
    if (PetscAbs(m) >= 1.0) {
        return M1Minus(m);
    } else {
        PetscReal beta = 0.125;
        return M2Minus(m) * (1.0 + 16.0 * beta * M2Plus(m));
    }
}
PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::P5Plus(PetscReal m, double fa) {
    if (PetscAbs(m) >= 1.0) {
        return (M1Plus(m) / (m + 1E-30));
    } else {
        // compute alpha
        double alpha = 3.0 / 16.0 * (-4.0 + 5 * fa * fa);

        return (M2Plus(m) * ((2.0 - m) - 16. * alpha * m * M2Minus(m)));
    }
}
PetscReal ablate::finiteVolume::processes::NPhaseNonconservativeRHS::P5Minus(PetscReal m, double fa) {
    if (PetscAbs(m) >= 1.0) {
        return (M1Minus(m) / (m + 1E-30));
    } else {
        double alpha = 3.0 / 16.0 * (-4.0 + 5 * fa * fa);
        return (M2Minus(m) * ((-2.0 - m) + 16. * alpha * m * M2Plus(m)));
    }
}

PetscErrorCode ablate::finiteVolume::processes::NPhaseNonconservativeRHS::ComputeNonconservativeRHS(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void *ctx) {
    PetscFunctionBegin;

    auto nPhaseNonconservativeRHSProcess = (NPhaseNonconservativeRHS *)ctx;

    // Get the subdomain from the solver - use non-const version since we need to modify it
    auto subDomain = const_cast<FiniteVolumeSolver&>(solver).GetSubDomainPtr();
    if (!subDomain) {
        throw std::runtime_error("SubDomain not set in solver");
    }

    // Get dimensions
    const PetscInt dim = subDomain->GetDimensions();

    // Get cell range
    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);

    // Get field offsets
    const auto &alphakField = subDomain->GetField(ALPHAK);
    const auto &alphakOffset = alphakField.offset;

    // Get arrays
    PetscScalar *flowArray;
    VecGetArray(locXVec, &flowArray) >> utilities::PetscUtilities::checkError;
    PetscScalar *fArray;
    VecGetArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;

    // Get aux fields
    subDomain->UpdateAuxLocalVector();
    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    PetscScalar *auxArray;
    VecGetArray(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;

    // Get field information
    const auto &debugfield = subDomain->GetField("debug");
    const auto &velocityField = subDomain->GetField(NPhaseFlowFields::UI);
    const auto &densityField = subDomain->GetField(NPhaseFlowFields::RHO);
    const auto &pressureField = subDomain->GetField(NPhaseFlowFields::PRESSURE);
    const auto &soskField = subDomain->GetField(NPhaseFlowFields::SOSK);

    // Verify debug field has enough components
    if (debugfield.numberComponents < nPhaseNonconservativeRHSProcess->nPhases) {
        throw std::runtime_error("Debug field must have at least " + std::to_string(nPhaseNonconservativeRHSProcess->nPhases) + " components to store terms for each phase");
    }

    // Compute boundary distances if not already done
    if (nPhaseNonconservativeRHSProcess->cellBoundaryDistance.empty()) {
        nPhaseNonconservativeRHSProcess->ComputeBoundaryDistances();
    }

    nPhaseNonconservativeRHSProcess->nPhases = alphakField.numberComponents;

    auto& cellValues = nPhaseNonconservativeRHSProcess->cellValues;
    const auto& cellToFaces = nPhaseNonconservativeRHSProcess->cellToFaces;
    const auto& faceToCells = nPhaseNonconservativeRHSProcess->faceToCells;
    auto& cellBoundaryDistance = nPhaseNonconservativeRHSProcess->cellBoundaryDistance;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt cell = cellRange.GetPoint(c);
        auto& cellVal = cellValues[cell];
        cellVal.alphak.resize(nPhaseNonconservativeRHSProcess->nPhases);
        cellVal.sosk.resize(nPhaseNonconservativeRHSProcess->nPhases);

        const PetscScalar *alphak;
        xDMPlexPointLocalRead(dm, cell, alphakField.id, flowArray, &alphak);
        PetscReal *rho, *p, *u, *sosk;
        xDMPlexPointLocalRead(auxDM, cell, densityField.id, auxArray, &rho);
        xDMPlexPointLocalRead(auxDM, cell, pressureField.id, auxArray, &p);
        xDMPlexPointLocalRead(auxDM, cell, velocityField.id, auxArray, &u);
        xDMPlexPointLocalRead(auxDM, cell, soskField.id, auxArray, &sosk);

        if (alphak) {
            for (PetscInt k = 0; k < nPhaseNonconservativeRHSProcess->nPhases; k++) {
                cellVal.alphak[k] = alphak[k];
                cellVal.sosk[k] = sosk[k];
            }
        }
        cellVal.rho = *rho;
        cellVal.p = *p;
        for (PetscInt d = 0; d < dim; d++) {
            cellVal.u[d] = u[d];
        }
        cellVal.divU = 0.0;
    }

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt cell = cellRange.GetPoint(c);
        auto cellValIt = cellValues.find(cell);
        if (cellValIt == cellValues.end()) continue;
        auto& cellVal = cellValIt->second;

        const auto bdIt = cellBoundaryDistance.find(cell);
        if (bdIt != cellBoundaryDistance.end() && bdIt->second <= 5) {
            cellVal.divU = 0.0;
            continue;
        }

        const auto cellFacesIt = cellToFaces.find(cell);
        if (cellFacesIt == cellToFaces.end()) continue;

        PetscReal ujj = 0.0;
        for (PetscInt face : cellFacesIt->second) {
            std::vector<PetscReal> faceNormal(dim);
            PetscReal faceSign = 1.0;
            PetscReal faceAreaMag = 1.0;
            const auto faceCellsIt = faceToCells.find(face);
            if (faceCellsIt == faceToCells.end()) continue;
            const auto& cells = faceCellsIt->second;
            if (dim == 1) {
                faceSign = (cells[0] == cell) ? 1.0 : -1.0;
                faceNormal[0] = faceSign;
            } else {
                DMPlexFaceCentroidOutwardAreaNormal(auxDM, cell, face, nullptr, faceNormal.data());
                faceAreaMag = MagVector(dim, faceNormal.data());
            }

            if (cells.size() != 2) {
                throw std::runtime_error("Face " + std::to_string(face) + " has " + std::to_string(cells.size()) + " cells, expected 2");
            }

            auto cellLIt = cellValues.find(cells[0]);
            auto cellRIt = cellValues.find(cells[1]);
            if (cellLIt == cellValues.end() || cellRIt == cellValues.end()) continue;
            const auto& cellL = cellLIt->second;
            const auto& cellR = cellRIt->second;

            PetscReal rho12 = 0.5 * (cellL.rho + cellR.rho);

            PetscReal sosL = 0.0, sosR = 0.0;
            for (PetscInt k = 0; k < nPhaseNonconservativeRHSProcess->nPhases; k++) {
                if (cellL.alphak[k] > NPhaseFlowFields::ALPHAK_FLOOR && cellL.sosk[k] > 0.0) {
                    sosL += cellL.alphak[k] / cellL.sosk[k];
                }
                if (cellR.alphak[k] > NPhaseFlowFields::ALPHAK_FLOOR && cellR.sosk[k] > 0.0) {
                    sosR += cellR.alphak[k] / cellR.sosk[k];
                }
            }
            sosL = 1.0/sosL;
            sosR = 1.0/sosR;
            PetscReal sos12 = 0.5 * (sosL + sosR);

            // Compute normal velocities
            PetscReal unL = 0.0, unR = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                unL += cellL.u[d] * faceNormal[d];
                unR += cellR.u[d] * faceNormal[d];
            }

            PetscReal mL = unL / sos12;
            PetscReal mR = unR / sos12;

            PetscReal mBar2 = (PetscSqr(unL) + PetscSqr(unR)) / (2.0 * PetscSqr(sos12));
            PetscReal fa = 1.0;

            if (nPhaseNonconservativeRHSProcess->mInf > 0) {
                PetscReal mInf2 = PetscSqr(nPhaseNonconservativeRHSProcess->mInf);
                PetscReal mO2 = PetscMin(1.0, PetscMax(mBar2, mInf2));
                PetscReal mO = PetscSqrtReal(mO2);
                fa = mO * (2.0 - mO);
            }

            PetscReal m12 = M4Plus(mL) + M4Minus(mR) - (Kp / fa) * PetscMax(1.0 - (sigma * mBar2), 0) * (cellR.p - cellL.p) / (rho12 * sos12 * sos12 * pgsAlpha * pgsAlpha);
            PetscReal vRiem = sos12 * m12;

            //if vriem is nan (due to being unable to be computed near the boundary), just set to zero
            if (PetscIsNanReal(vRiem)) {
                vRiem = 0.0;
            }

            ujj += vRiem * faceAreaMag * faceSign;
        }

        cellVal.divU = ujj;

        // Update the debug field with separate terms for each phase
        PetscScalar *term;
        xDMPlexPointLocalRef(auxDM, cell, debugfield.id, auxArray, &term);
        PetscReal cellVolume;
        for (PetscInt k = 0; k < nPhaseNonconservativeRHSProcess->nPhases; k++) {
            if (term) {
            term[k] = cellVal.alphak[k] * cellVal.divU;
            }
                // Scale by cell size
                if (dim == 1) {
                    DMPlexGetMinRadius(auxDM, &cellVolume) >> utilities::PetscUtilities::checkError;
                    cellVolume *= 2.0;
                }
                if (dim > 1) {

                    DMPlexComputeCellGeometryFVM(auxDM, cell, &cellVolume, nullptr, nullptr) >> utilities::PetscUtilities::checkError;
                }

                //if term is greater than petscsmall, print it

        }

        // Update the solution vector for all phases
        PetscScalar *allFields = nullptr;
        DMPlexPointLocalRef(dm, cell, fArray, &allFields) >> utilities::PetscUtilities::checkError;
        if (allFields) {
            // Add the nonconservative term to each phase's equation
            for (PetscInt k = 0; k < nPhaseNonconservativeRHSProcess->nPhases; k++) {
                allFields[alphakOffset + k] += cellVal.alphak[k] * cellVal.divU / cellVolume;
            }
        }
    }

    // Clean up
    VecRestoreArray(locXVec, &flowArray);
    VecRestoreArray(locFVec, &fArray);
    VecRestoreArray(auxVec, &auxArray);
    solver.RestoreRange(cellRange);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::NPhaseNonconservativeRHS::EnforceAlphaKBounds(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locXVec, void *ctx) {
    PetscFunctionBegin;

    // Get the subdomain from the solver
    auto subDomain = const_cast<FiniteVolumeSolver&>(solver).GetSubDomainPtr();
    if (!subDomain) {
        throw std::runtime_error("SubDomain not set in solver");
    }

    // Get cell range
    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);

    // Get field information
    const auto &alphakField = subDomain->GetField(ALPHAK);
    const auto &alphakOffset = alphakField.offset;

    // Get array
    PetscScalar *xArray;
    VecGetArray(locXVec, &xArray) >> utilities::PetscUtilities::checkError;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt cell = cellRange.GetPoint(c);
        PetscScalar *allFields = nullptr;
        DMPlexPointLocalRef(dm, cell, xArray, &allFields) >> utilities::PetscUtilities::checkError;

        if (allFields) {
            // First enforce minimum bound of 0
            for (PetscInt k = 0; k < alphakField.numberComponents; k++) {
                allFields[alphakOffset + k] = PetscMax(0.0, allFields[alphakOffset + k]);
            }

            // Compute sum of alpha values
            PetscReal alphaSum = 0.0;
            for (PetscInt k = 0; k < alphakField.numberComponents; k++) {
                alphaSum += allFields[alphakOffset + k];
            }

            // Normalize if sum is not equal to 1 (either greater than or less than)
            if (PetscAbs(alphaSum - 1.0) > PETSC_SMALL) {
                PetscReal normalizationFactor = 1.0 / alphaSum;
                for (PetscInt k = 0; k < alphakField.numberComponents; k++) {
                    allFields[alphakOffset + k] *= normalizationFactor;
                }
            }
        }
    }

    // Clean up
    VecRestoreArray(locXVec, &xArray);
    solver.RestoreRange(cellRange);

    PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::NPhaseNonconservativeRHS, "calculates nonconservative rhs term",
         OPT(double, "mInf", "must be same as mInf in ausmpUp"),
         OPT(ablate::finiteVolume::processes::PressureGradientScaling, "pgs", "must be same as pgs in ausmpUp"));
