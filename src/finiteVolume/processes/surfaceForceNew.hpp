#ifndef ABLATELIBRARY_FINITEVOLUME_SURFACEFORCENEW_HPP
#define ABLATELIBRARY_FINITEVOLUME_SURFACEFORCENEW_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "domain/reverseRange.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "finiteVolume/stencils/gaussianConvolution.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "twoPhaseEulerAdvection.hpp"



namespace ablate::finiteVolume::processes {

class SurfaceForceNew : public Process {

  private:

    PetscReal sigma;

    void GetFieldVectors(const ablate::domain::SubDomain& subDomain, Vec *subLocalVec, Vec *subGlobalVec);

    DM subDM = nullptr;
    IS subIS = nullptr;
    enum VecLoc { LOCAL, GLOBAL};

    ablate::domain::Range cellRange = {};
    ablate::domain::ReverseRange reverseCellRange = {};

    void ClearData();
    std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> cellGaussianConv = nullptr;

  public:
    /**
    *
    * @param Gamma
    * @param sigma
    */
    explicit SurfaceForceNew(const PetscReal sigma = 0);

    /**
    * Clean up the dm created
    */
    ~SurfaceForceNew() override;

    /**
    * Setup the process to define the vertex dm
    * @param flow
    */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    static PetscErrorCode ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx);
};

}  // namespace ablate::finiteVolume::processes
#endif
