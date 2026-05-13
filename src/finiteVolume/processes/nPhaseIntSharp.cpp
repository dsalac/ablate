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
#include "utilities/petscUtilities.hpp"
#include <petsc/private/dmpleximpl.h>

namespace ablate::finiteVolume::processes {

    void ablate::finiteVolume::processes::NPhaseIntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
        NPhaseIntSharp::subDomain = solver.GetSubDomainPtr();
    }

    ablate::finiteVolume::processes::NPhaseIntSharp::Form
    ablate::finiteVolume::processes::NPhaseIntSharp::ParseForm(const std::string &raw) {
        std::string s;
        s.reserve(raw.size());
        for (char c : raw) {
            if (c == '_' || c == '-' || c == ' ') continue;
            s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (s.empty() || s == "parameswaranmandal" || s == "pm" || s == "default") {
            return Form::PARAMESWARAN_MANDAL;
        }
        if (s == "chiulin" || s == "cl" || s == "conservative" || s == "div") {
            return Form::CHIU_LIN;
        }
        throw std::invalid_argument("NPhaseIntSharp: unknown form '" + raw +
                                    "' (expected 'parameswaran_mandal' or 'chiu_lin')");
    }

    ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharp(const std::vector<PetscReal>& Gammak, const std::vector<PetscReal>& epsilonk, const std::vector<PetscInt>& flipPhiTildek, PetscReal boundaryLayerMultiplier, std::string formIn) : Gammak(Gammak), epsilonk(epsilonk), flipPhiTildek(flipPhiTildek), form(ParseForm(formIn)), boundaryLayerMultiplier(boundaryLayerMultiplier) {
        // Initialize boundary layer thickness as a multiple of minRadius (will be set in Setup)
        boundaryLayerThickness = 0.0;
        minRadius = 0.0;
        for (int i = 0; i < 6; ++i) boundingBox[i] = 0.0;
    }

    ablate::finiteVolume::processes::NPhaseIntSharp::~NPhaseIntSharp() {
        DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError;
        if (fluxDM) DMDestroy(&fluxDM) >> utilities::PetscUtilities::checkError;
    }

    void ablate::finiteVolume::processes::NPhaseIntSharp::EnsureFluxDM(DM dm, PetscInt phases) {
        if (fluxDM) return;

        PetscInt dim;
        DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

        DMClone(dm, &fluxDM) >> utilities::PetscUtilities::checkError;

        DM coordDM = nullptr;
        DMGetCoordinateDM(dm, &coordDM) >> utilities::PetscUtilities::checkError;
        DMSetCoordinateDM(fluxDM, coordDM) >> utilities::PetscUtilities::checkError;

        PetscInt cStart, cEnd;
        DMPlexGetHeightStratum(fluxDM, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

        PetscSection section;
        PetscSectionCreate(PetscObjectComm((PetscObject)fluxDM), &section) >> utilities::PetscUtilities::checkError;
        PetscSectionSetChart(section, cStart, cEnd) >> utilities::PetscUtilities::checkError;
        const PetscInt dofPerCell = dim * phases;
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscSectionSetDof(section, c, dofPerCell) >> utilities::PetscUtilities::checkError;
        }
        PetscSectionSetUp(section) >> utilities::PetscUtilities::checkError;
        DMSetLocalSection(fluxDM, section) >> utilities::PetscUtilities::checkError;
        PetscSectionDestroy(&section) >> utilities::PetscUtilities::checkError;
        DMSetUp(fluxDM) >> utilities::PetscUtilities::checkError;

        Vec cellGeom = nullptr, faceGeom = nullptr;
        DMPlexComputeGeometryFVM(fluxDM, &cellGeom, &faceGeom) >> utilities::PetscUtilities::checkError;
        if (cellGeom) VecDestroy(&cellGeom) >> utilities::PetscUtilities::checkError;
        if (faceGeom) VecDestroy(&faceGeom) >> utilities::PetscUtilities::checkError;
    }

    void ablate::finiteVolume::processes::NPhaseIntSharp::ComputeBoundaryInformation(DM dm) {
        PetscInt dim;
        DMGetDimension(dm, &dim);

        // Get bounding box of the domain
        PetscReal xymin[3], xymax[3];
        DMGetBoundingBox(dm, xymin, xymax);

        // Store bounding box in the format [xmin, xmax, ymin, ymax, zmin, zmax]
        boundingBox[0] = xymin[0];  // xmin
        boundingBox[1] = xymax[0];  // xmax
        boundingBox[2] = xymin[1];  // ymin
        boundingBox[3] = xymax[1];  // ymax
        boundingBox[4] = xymin[2];  // zmin
        boundingBox[5] = xymax[2];  // zmax

        // Get minimum radius (characteristic mesh size)
        DMPlexGetMinRadius(dm, &minRadius);

        // Set boundary layer thickness as a multiple of minRadius (e.g., 3-5 cell layers)
        boundaryLayerThickness = boundaryLayerMultiplier * minRadius;

        PetscInt cStart, cEnd;
        DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);

        DMLabel ghostLabel = nullptr;
        DMGetLabel(dm, "ghost", &ghostLabel);

        // Compute boundary distance and weight for each cell
        for (PetscInt cell = cStart; cell < cEnd; ++cell) {
            if (ghostLabel) {
                PetscInt ghostVal = -1;
                DMLabelGetValue(ghostLabel, cell, &ghostVal);
                if (ghostVal >= 0) {
                    continue;
                }
            }
            PetscReal centroid[3];
            DMPlexPointGeometricData(dm, cell, nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;

            // Compute minimum distance to any boundary
            PetscReal minDistToBoundary = PETSC_INFINITY;

            // Check distance to each boundary face
            for (int d = 0; d < dim; ++d) {
                // Distance to lower boundary
                PetscReal distToLower = centroid[d] - boundingBox[2*d];
                if (distToLower < minDistToBoundary) {
                    minDistToBoundary = distToLower;
                }

                // Distance to upper boundary
                PetscReal distToUpper = boundingBox[2*d + 1] - centroid[d];
                if (distToUpper < minDistToBoundary) {
                    minDistToBoundary = distToUpper;
                }
            }

            cellBoundaryDistances[cell] = minDistToBoundary;

            // Compute boundary weight: 1.0 for interior, 0.0 for boundary (binary)
            PetscReal weight = (minDistToBoundary >= boundaryLayerThickness) ? 1.0 : 0.0;
            cellBoundaryWeights[cell] = weight;
        }
    }

    PetscReal ablate::finiteVolume::processes::NPhaseIntSharp::GetBoundaryWeight(PetscInt cell) const {
        auto it = cellBoundaryWeights.find(cell);
        if (it != cellBoundaryWeights.end()) {
            return it->second;
        }
        return 0.0;
    }

    void nPhaseIntSharpPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::NPhaseIntSharp* nPhaseIntSharpProcess) {
        nPhaseIntSharpProcess->PreStage(flowTs, solver, stagetime);
    }



    void NPhaseIntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {

        NPhaseIntSharp::subDomain = flow.GetSubDomainPtr();

        auto dim = flow.GetSubDomain().GetDimensions();
        auto dm = flow.GetSubDomain().GetDM();

        PetscFE fe_coords;
        PetscInt k = 1;

        DMClone(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
        PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
        DMSetField(vertexDM, 0, nullptr, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
        PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;

        DMCreateDS(vertexDM) >> utilities::PetscUtilities::checkError;
        ComputeBoundaryInformation(dm);

        auto fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver*>(&flow);

        if (!fvSolver) {
          return;
        }

        fvSolver->EnableSlopeLimiterFor(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
        fvSolver->EnableSlopeLimiterFor(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);

        auto nPhaseIntSharpPreStage = std::bind(nPhaseIntSharpPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
        flow.RegisterPreStage(nPhaseIntSharpPreStage);
    }

    PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
        PetscFunctionBegin;

        const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);

        ablate::domain::Range cellRange;
        fvSolver.GetCellRangeWithoutGhost(cellRange);

        PetscInt dim;
        PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));

        DM dm = fvSolver.GetSubDomain().GetDM();

        Vec globFlowVec;
        PetscCall(TSGetSolution(flowTs, &globFlowVec));

        PetscScalar *flowArray;
        PetscCall(VecGetArray(globFlowVec, &flowArray));

        Vec locFVec;
        PetscCall(DMGetLocalVector(dm, &locFVec));
        PetscCall(VecZeroEntries(locFVec));

        Vec locX = solver.GetSubDomain().GetSolutionVector();

        ablate::finiteVolume::processes::NPhaseIntSharp *process = this;

        std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;

        DM auxDM = subDomain->GetAuxDM();
        Vec auxVec = subDomain->GetAuxVector();

        Vec vertexVec;
        DMGetLocalVector(process->vertexDM, &vertexVec);

        const PetscScalar *solArray;
        PetscScalar *auxArray;
        PetscScalar *vertexArray;
        PetscScalar *fArray;

        VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
        VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
        VecGetArray(vertexVec, &vertexArray);
        VecGetArray(locFVec, &fArray);

        const auto &alphakField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
        const auto &alphakrhokField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
        const auto &fsharpkField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::FSHARPK);

        std::size_t phases = alphakField.numberComponents;

        if (form == Form::CHIU_LIN) {
          printf("Chiu-Liu form needs to be checked.\n");
          printf("%s::%d\n", __FILE__, __LINE__);
          exit(0);
            for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
                const PetscInt cell = cellRange.GetPoint(c);
                PetscScalar *fsharpkCell = nullptr;
                xDMPlexPointLocalRef(auxDM, cell, fsharpkField.id, auxArray, &fsharpkCell);
                if (fsharpkCell) {
                    for (std::size_t k = 0; k < phases; ++k) fsharpkCell[k] = 0.0;
                }
            }

            PetscInt fStart = 0, fEnd = 0;
            DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;

            DMLabel ghostLabel = nullptr;
            DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

            for (PetscInt face = fStart; face < fEnd; ++face) {
                if (ghostLabel) {
                    PetscInt gv = -1;
                    DMLabelGetValue(ghostLabel, face, &gv) >> utilities::PetscUtilities::checkError;
                    if (gv > 0) continue;
                }

                PetscInt supSize = 0;
                const PetscInt *support = nullptr;
                DMPlexGetSupportSize(dm, face, &supSize) >> utilities::PetscUtilities::checkError;
                DMPlexGetSupport(dm, face, &support) >> utilities::PetscUtilities::checkError;
                if (supSize != 2) continue; // physical boundary face: zero-flux Neumann
                const PetscInt L = support[0];
                const PetscInt R = support[1];

                PetscReal centroidL[3] = {0.0, 0.0, 0.0};
                PetscReal centroidR[3] = {0.0, 0.0, 0.0};
                PetscReal volL = 0.0;
                PetscReal volR = 0.0;
                DMPlexComputeCellGeometryFVM(dm, L, &volL, centroidL, nullptr) >> utilities::PetscUtilities::checkError;
                DMPlexComputeCellGeometryFVM(dm, R, &volR, centroidR, nullptr) >> utilities::PetscUtilities::checkError;
                if (volL <= 0.0 || volR <= 0.0) continue;

                PetscReal areaFace = 0.0;
                PetscReal centroidF[3] = {0.0, 0.0, 0.0};
                PetscReal normalF[3]   = {0.0, 0.0, 0.0};
                DMPlexComputeCellGeometryFVM(dm, face, &areaFace, centroidF, normalF) >> utilities::PetscUtilities::checkError;

                PetscReal dn = 0.0;
                for (PetscInt d = 0; d < dim; ++d) {
                    PetscReal dxd = centroidR[d] - centroidL[d];
                    dn += dxd * dxd;
                }
                dn = PetscSqrtReal(dn);
                if (dn <= ablate::utilities::Constants::tiny) continue;

                const PetscScalar *alphakL = nullptr;
                const PetscScalar *alphakR = nullptr;
                xDMPlexPointLocalRead(dm, L, alphakField.id, solArray, &alphakL);
                xDMPlexPointLocalRead(dm, R, alphakField.id, solArray, &alphakR);
                if (!alphakL || !alphakR) continue;

                PetscScalar *fsharpL = nullptr;
                PetscScalar *fsharpR = nullptr;
                xDMPlexPointLocalRef(auxDM, L, fsharpkField.id, auxArray, &fsharpL);
                xDMPlexPointLocalRef(auxDM, R, fsharpkField.id, auxArray, &fsharpR);

                const PetscReal wL = process->GetBoundaryWeight(L);
                const PetscReal wR = process->GetBoundaryWeight(R);

                for (std::size_t k = 0; k < phases; ++k) {
                    const PetscReal aL = alphakL[k];
                    const PetscReal aR = alphakR[k];

                    if ((aL <= 1e-3 && aR <= 1e-3) || (aL >= 1.0 - 1e-3 && aR >= 1.0 - 1e-3)) {
                        continue;
                    }

                    const PetscReal af  = 0.5 * (aL + aR);
                    const PetscReal dad = (aR - aL) / dn;             // normal gradient (FD)
                    const PetscReal sg  = (aR > aL) ? 1.0 : (aR < aL ? -1.0 : 0.0);

                    PetscReal aftilde = af;
                    if (process->flipPhiTildek[k] == 1) aftilde = 1.0 - af;

                    const PetscReal Gk = process->Gammak[k];
                    const PetscReal Ek = process->epsilonk[k];

                    const PetscReal F_dot_n = Gk * Ek * dad - Gk * aftilde * (1.0 - aftilde) * sg;
                    const PetscReal flux_LR = F_dot_n * areaFace;     // signed mass flux across face

                    if (fsharpL && wL >= 0.5) fsharpL[k] += flux_LR / volL;
                    if (fsharpR && wR >= 0.5) fsharpR[k] -= flux_LR / volR;
                }
            }
        }

        for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
            const PetscInt cell = cellRange.GetPoint(c);

            //keep old values
            std::vector<PetscReal> alphakold(phases);
            std::vector<PetscReal> alphakrhokold(phases);
            PetscReal rhoold = 0.0;
            std::vector<PetscReal> uiold(dim);
            std::vector<PetscReal> rhokold(phases);


            PetscScalar *allFields = nullptr;
            DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;

            //coordinates of this cell:
            PetscReal centroid[3];
            DMPlexPointGeometricData(dm, cell, nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;

            // Get field pointers ONCE outside the loop
            const PetscScalar *alphakFieldPtr;
            const PetscScalar *alphakrhokFieldPtr;
            xDMPlexPointLocalRead(dm, cell, alphakField.id, solArray, &alphakFieldPtr);
            xDMPlexPointLocalRead(dm, cell, alphakrhokField.id, solArray, &alphakrhokFieldPtr);

            // Now access the components correctly
            for (std::size_t k = 0; k < phases; ++k) {
                alphakold[k] = alphakFieldPtr[k];
                alphakrhokold[k] = alphakrhokFieldPtr[k];
                rhoold += alphakrhokFieldPtr[k];

                // Avoid division by zero
                if (alphakFieldPtr[k] > PETSC_SMALL) {
                    rhokold[k] = alphakrhokFieldPtr[k] / alphakFieldPtr[k];
                } else {
                    rhokold[k] = 0.0;
                }
            }


            // Check for zero density to avoid division by zero
            if (rhoold < PETSC_SMALL) continue;  // Skip this cell entirely

            for (PetscInt d = 0; d < dim; ++d) uiold[d] = allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / rhoold;

            //compute fsharpk for all k
            for (std::size_t k = 0; k < phases; ++k) {
                // Use the field pointers we already got, or get them if needed
                PetscScalar *fsharpk;
                xDMPlexPointLocalRef(auxDM, cell, fsharpkField.id, auxArray, &fsharpk);

                if (alphakold[k] <= 1e-3 || alphakold[k] >= 1.0 - 1e-3) {
                    fsharpk[k] = 0.0;
                    continue;
                }

                PetscReal Gammak = process->Gammak[k];
                PetscReal epsilonk = process->epsilonk[k];

                PetscReal alphaktilde = alphakold[k];  // Use the stored old value
                if (process->flipPhiTildek[k] == 1) {
                    alphaktilde = 1 - alphakold[k];
                }

                if (form == Form::PARAMESWARAN_MANDAL) {
                    PetscScalar gradalphak[3];
                    DMPlexCellGradFromCell(auxDM, cell, auxVec, alphakField.id, k, gradalphak) >> utilities::PetscUtilities::checkError;

                    PetscReal normgradalphak = 0.0;
                    for (PetscInt d = 0; d < dim; ++d) {
                        normgradalphak += PetscSqr(gradalphak[d]);
                    }
                    normgradalphak = PetscSqrtReal(normgradalphak);

                    fsharpk[k] = Gammak * ( -alphaktilde * (1 - alphaktilde) * (1 - 2 * alphaktilde) + epsilonk * (1 - 2 * alphaktilde) * normgradalphak );
                }
            }

            PetscReal v2_old = 0.0;
            for (PetscInt d = 0; d < dim; ++d) v2_old += uiold[d] * uiold[d];
            PetscReal rhoe_old = allFields[ablate::finiteVolume::NPhaseFlowFields::RHOE];
            PetscReal e_old = (rhoe_old - 0.5 * rhoold * v2_old) / rhoold;

            PetscReal rhok_fallback = rhoold;

            for (std::size_t k = 0; k < phases; ++k) {
                PetscScalar *fsharpk;
                xDMPlexPointLocalRef(auxDM, cell, fsharpkField.id, auxArray, &fsharpk);
                allFields[alphakField.offset + k] += fsharpk[k];
                if (allFields[alphakField.offset + k] < 0.0)      allFields[alphakField.offset + k] = 0.0;
                else if (allFields[alphakField.offset + k] > 1.0) allFields[alphakField.offset + k] = 1.0;
            }

            PetscReal alphasum = 0.0;
            for (std::size_t k = 0; k < phases; ++k) alphasum += allFields[alphakField.offset + k];
            if (alphasum > PETSC_SMALL) {
                PetscReal invsum = 1.0 / alphasum;
                for (std::size_t k = 0; k < phases; ++k) allFields[alphakField.offset + k] *= invsum;
            }

            PetscReal rho = 0.0;
            for (std::size_t k = 0; k < phases; ++k) {
                PetscReal rhok_k = (rhokold[k] > PETSC_SMALL) ? rhokold[k] : rhok_fallback;
                allFields[alphakrhokField.offset + k] = allFields[alphakField.offset + k] * rhok_k;
                rho += allFields[alphakrhokField.offset + k];
            }

            for (PetscInt d = 0; d < dim; ++d) {
                allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] = rho * uiold[d];
            }
            allFields[ablate::finiteVolume::NPhaseFlowFields::RHOE] = rho * (e_old + 0.5 * v2_old);
        }

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to restore arrays\n");
        VecRestoreArrayRead(locX, &solArray);
        VecRestoreArray(auxVec, &auxArray);
        VecRestoreArray(vertexVec, &vertexArray);
        VecRestoreArray(locFVec, &fArray);
        solver.RestoreRange(cellRange);

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to restore vertex vector\n");
        DMRestoreLocalVector(process->vertexDM, &vertexVec);
        VecDestroy(&vertexVec);

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] PreStage completed successfully\n");
        PetscFunctionReturn(0);
    }

}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process,
    ablate::finiteVolume::processes::NPhaseIntSharp,
    "N-phase interface regularization term",
    ARG(std::vector<PetscReal>, "Gammak", "Gamma, velocity scale parameter (approx. umax)"),
    ARG(std::vector<PetscReal>, "epsilonk", "epsilon, interface thickness scale parameter (approx. h)"),
    ARG(std::vector<PetscInt>, "flipPhiTildek", "if 1: phiTilde-->1-phiTilde, if 0: keep phiTilde (set to 1 if primary phase is phi=0 or 0 if phi=1)"),
    ARG(PetscReal, "boundaryLayerMultiplier", "multiplier for boundary layer thickness (default: 3.0)"),
    OPT(std::string, "form", "discrete sharpening form: 'parameswaran_mandal' (default; cell-centered scalar) or 'chiu_lin' (conservative divergence)"));
