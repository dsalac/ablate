#ifndef ABLATELIBRARY_RBF_intMQ_HPP
#define ABLATELIBRARY_RBF_intMQ_HPP

#include "rbf.hpp"

#define __RBF_intMQ_DEFAULT_PARAM 0.1

namespace ablate::domain::rbf {

// Integrated MQ
class intMQ : virtual public RBF {
   private:
    const double scale = -1;
    double a[5] = {1.0, 0.0, 0.0, 0.0, 0.0};

   public:
    std::string_view type() const override { return "intMQ"; }

    intMQ(int p = 4, double scale = 0.1, bool doesNotHaveDerivatives = false, bool doesNotHaveInterpolation = false, bool returnNeighborVertices = false);

    PetscReal RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) override;
    PetscReal RBFDer(PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override;
};

}  // namespace ablate::domain::rbf

#endif  // ABLATELIBRARY_RBF_intMQ_HPP
