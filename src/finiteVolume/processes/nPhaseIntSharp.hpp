#ifndef ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP
#define ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP

#include <petsc.h>
#include <memory>
#include <string>
#include <vector>
#include "domain/range.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
// #include "twoPhaseEulerAdvection.hpp"
#include "nPhaseAllaireAdvection.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseIntSharp : public Process {

   public:
    enum class Form { PARAMESWARAN_MANDAL, CHIU_LIN };

   private:
    //mesh for vertex information
    DM vertexDM{};
    DM fluxDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;
    std::vector<PetscReal> Gammak;
    std::vector<PetscReal> epsilonk;
    std::vector<PetscInt> flipPhiTildek;
    Form form;

    PetscReal boundingBox[6];  // [xmin, xmax, ymin, ymax, zmin, zmax]
    PetscReal minRadius;
    PetscReal boundaryLayerThickness;
    PetscReal boundaryLayerMultiplier;
    std::map<PetscInt, PetscReal> cellBoundaryDistances;
    std::map<PetscInt, PetscReal> cellBoundaryWeights;

    void ComputeBoundaryInformation(DM dm);

    void EnsureFluxDM(DM dm, PetscInt phases);

    /**
     * Get boundary weight for a cell (1.0 for interior, 0.0 for boundary, smooth transition in between)
     * @param cell The cell index
     * @return Boundary weight between 0.0 and 1.0
     */
    PetscReal GetBoundaryWeight(PetscInt cell) const;

    static Form ParseForm(const std::string &s);

   public:

    explicit NPhaseIntSharp(
        const std::vector<PetscReal>& Gammak,
        const std::vector<PetscReal>& epsilonk,
        const std::vector<PetscInt>& flipPhiTildek,
        PetscReal boundaryLayerMultiplier = 3.0,
        std::string form = "parameswaran_mandal");

    ~NPhaseIntSharp() override;

    PetscErrorCode PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime);

    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;


};
}  // namespace ablate::finiteVolume::processes
#endif
