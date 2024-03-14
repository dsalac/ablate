#ifndef ABLATELIBRARY_INTERFACERECONSTRUCTION_HPP
#define ABLATELIBRARY_INTERFACERECONSTRUCTION_HPP

#include "domain/RBF/ga.hpp"
#include "domain/RBF/hybrid.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/phs.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/subDomain.hpp"



#include "domain/range.hpp"
#include "domain/reverseRange.hpp"



namespace ablate::levelSet {

  class Reconstruction {

    private:

      std::shared_ptr<ablate::domain::rbf::RBF> vertRBF = nullptr;
      std::shared_ptr<ablate::domain::rbf::RBF> cellRBF = nullptr;

      std::shared_ptr<ablate::domain::SubDomain> subDomain = nullptr;


      void CurvatureViaGaussian(DM dm, const PetscInt cell, const Vec vec, const ablate::domain::Field *lsField, double *H);

      void CutCellLevelSetValues(std::shared_ptr<ablate::domain::SubDomain> subDomain, ablate::domain::Range cellRange, ablate::domain::Range vertRange, ablate::domain::ReverseRange reverseVertRange, const PetscInt *cellMask, DM solDM, Vec solVec, const PetscInt vofID, DM auxDM, Vec auxVec, const PetscInt normalID, const PetscInt lsID);

      void VertexUpwindGrad(DM dm, PetscScalar *gradArray, const PetscInt gradID, const PetscInt v, const PetscReal direction, PetscReal *g);


    public:
      Reconstruction(std::shared_ptr<ablate::domain::SubDomain> subDomain);
      ~Reconstruction();

      void ComputeCurvature(const Vec solVec, const ablate::domain::Field *vofField, const PetscInt nLevels, const ablate::domain::Field *lsField, const ablate::domain::Field *vertexNormalField, const ablate::domain::Field *cellNormalField, const ablate::domain::Field *curvField);

  };


}  // namespace ablate::levelSet::Reconstruction
#endif  // ABLATELIBRARY_INTERFACERECONSTRUCTION_HPP



//namespace ablate::finiteVolume::processes {

//  class SurfaceForce : public Process {


//    private:

//    PetscReal sigma;



//    public:


//    explicit SurfaceForce(PetscReal sigma);


//    ~SurfaceForce() override;

//    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
//    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

//    /**
//     * static function private function to compute surface force and add source to eulerset
//     * @param solver
//     * @param dm - DM of the cell-centered data
//     * @param time - Current time of data in locX
//     * @param locX - Local vector containing current solution
//     * @param fVec - Vector to store the Cell-centered body-force
//     * @param ctx - Pointer to ablate::finiteVolume::processes::SurfaceForce
//     * @return
//     */
//    static PetscErrorCode ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx);
//  };
//}  // namespace ablate::finiteVolume::processes
//#endif
