# ablate::parameters::Parameters
## ablate::parameters::FactoryParameters*
Creates a parameter list based upon a factory.  Should be default for factory parsing.

# ablate::finiteVolume::fieldFunctions::CompressibleFlowState
## ablate::finiteVolume::fieldFunctions::CompressibleFlowState*
a simple structure used to describe a compressible flow field using an EOS, T, pressure, vel, Yi

eos (req) 
: (ablate::eos::EOS) the eos used for the flow field

temperature (req) 
: (ablate::mathFunctions::MathFunction) the temperature field (K)

pressure (req) 
: (ablate::mathFunctions::MathFunction) the pressure field (Pa)

velocity (req) 
: (ablate::mathFunctions::MathFunction) the velocity field (m/2)

massFractions
: (ablate::mathFunctions::FieldFunction) a fieldFunctions used to describe all mass fractions

# ablate::finiteVolume::fluxCalculator::FluxCalculator
## ablate::finiteVolume::fluxCalculator::Ausm
AUSM Flux Spliting: "A New Flux Splitting Scheme" Liou and Steffen, pg 26, Eqn (6), 1993

## ablate::finiteVolume::fluxCalculator::AverageFlux
Takes the average of the left/right faces.  Only useful for debugging.

## ablate::finiteVolume::fluxCalculator::OffFlux
Turns of convective flux through the face.

## ablate::finiteVolume::fluxCalculator::AusmpUp
A sequel to AUSM, Part II: AUSM+-up for all speeds, Meng-Sing Liou, Pages 137-170, 2006

mInf
: (double) the reference mach number

pgs
: (ablate::finiteVolume::processes::PressureGradientScaling) Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows

## ablate::finiteVolume::fluxCalculator::Rieman
Exact Rieman Solution

eos (req) 
: (ablate::eos::EOS) only valid for perfect gas

## ablate::finiteVolume::fluxCalculator::Riemann2Gas
Exact Riemann Solution for 2 Perfect Gasses

eosL (req) 
: (ablate::eos::EOS) only valid for perfect gas

eosR (req) 
: (ablate::eos::EOS) only valid for perfect gas

## ablate::finiteVolume::fluxCalculator::RiemannStiff
Exact Riemann Solution for 2 Stiffened Gasses

eosL (req) 
: (ablate::eos::EOS) only valid for perfect or stiffened gas

eosR (req) 
: (ablate::eos::EOS) only valid for perfect or stiffened gas

# ablate::finiteVolume::boundaryConditions::BoundaryCondition
## ablate::finiteVolume::boundaryConditions::EssentialGhost
essential (Dirichlet condition) for ghost cell based boundaries

boundaryName (req) 
: (string) the name for this boundary condition

labelIds (req) 
: (int list) the ids on the mesh to apply the boundary condition

boundaryValue (req) 
: (ablate::mathFunctions::FieldFunction) the field function used to describe the boundary

labelName
: (string) the mesh label holding the boundary ids (default Face Sets)

# ablate::finiteVolume::processes::PressureGradientScaling
## ablate::finiteVolume::processes::PressureGradientScaling*
Rescales the thermodynamic pressure gradient scaling the acoustic propagation speeds to allow for a larger time step.

eos (req) 
: (ablate::eos::EOS) the equation of state used for the flow

alphaInit (req) 
: (double) the initial alpha

domainLength (req) 
: (double) the reference length of the domain

maxAlphaAllowed
: (double) the maximum allowed alpha during the simulation (default 100)

maxDeltaPressureFac
: (double) max variation from mean pressure (default 0.05)

log
: (ablate::monitors::logs::Log) where to record log (default is stdout)

# ablate::finiteVolume::processes::Process
## ablate::finiteVolume::processes::EulerTransport
build advection/diffusion for the euler field

parameters
: (ablate::parameters::Parameters) the parameters used by advection

eos (req) 
: (ablate::eos::EOS) the equation of state used to describe the flow

fluxCalculator
: (ablate::finiteVolume::fluxCalculator::FluxCalculator) the flux calculator (default is no advection)

transport
: (ablate::eos::transport::TransportModel) the diffusion transport model (default is no diffusion)

## ablate::finiteVolume::processes::SpeciesTransport
diffusion/advection for the species yi field

eos (req) 
: (ablate::eos::EOS) the equation of state used to describe the flow

fluxCalculator
: (ablate::finiteVolume::fluxCalculator::FluxCalculator) the flux calculator (default is no advection)

transport
: (ablate::eos::transport::TransportModel) the diffusion transport model (default is no diffusion)

## ablate::finiteVolume::processes::EVTransport
diffusion/advection for the specified EV

conserved (req) 
: (string) the name of the conserved (density*ev) of the variable

nonConserved (req) 
: (string) the name of the non-conserved (ev) of the variable

eos (req) 
: (ablate::eos::EOS) the equation of state used to describe the flow

fluxCalculator
: (ablate::finiteVolume::fluxCalculator::FluxCalculator) the flux calculator (default is no advection)

transport
: (ablate::eos::transport::TransportModel) the diffusion transport model (default is no diffusion)

## ablate::finiteVolume::processes::TChemReactions
reactions using the TChem v1 library

eos (req) 
: (ablate::eos::EOS) the tChem v1 eos

options
: (ablate::parameters::Parameters) any PETSc options for the chemistry ts

inertSpecies
: (string list) fix the Jacobian for any undetermined inertSpecies

massFractionBounds
: (double list) sets the minimum/maximum mass fraction passed to TChem Library. Must be a vector of size two [min,max] (default is no bounds)

## ablate::finiteVolume::processes::TwoPhaseEulerAdvection


eosGas (req) 
: (ablate::eos::EOS) 

eosLiquid (req) 
: (ablate::eos::EOS) 

fluxCalculatorGasGas (req) 
: (ablate::finiteVolume::fluxCalculator::FluxCalculator) 

fluxCalculatorGasLiquid (req) 
: (ablate::finiteVolume::fluxCalculator::FluxCalculator) 

fluxCalculatorLiquidGas (req) 
: (ablate::finiteVolume::fluxCalculator::FluxCalculator) 

fluxCalculatorLiquidLiquid (req) 
: (ablate::finiteVolume::fluxCalculator::FluxCalculator) 

## ablate::finiteVolume::processes::Gravity
build advection/diffusion for the euler field

vector (req) 
: (double list) gravitational acceleration vector

## ablate::finiteVolume::processes::PressureGradientScaling
Rescales the thermodynamic pressure gradient scaling the acoustic propagation speeds to allow for a larger time step.

eos (req) 
: (ablate::eos::EOS) the equation of state used for the flow

alphaInit (req) 
: (double) the initial alpha

domainLength (req) 
: (double) the reference length of the domain

maxAlphaAllowed
: (double) the maximum allowed alpha during the simulation (default 100)

maxDeltaPressureFac
: (double) max variation from mean pressure (default 0.05)

log
: (ablate::monitors::logs::Log) where to record log (default is stdout)

## ablate::finiteVolume::processes::ArbitrarySource
uses math functions to add arbitrary sources to the fvm method

## ablate::finiteVolume::processes::Buoyancy
build advection/diffusion for the euler field

vector (req) 
: (double list) gravitational acceleration vector

# ablate::finiteElement::boundaryConditions::BoundaryCondition
## ablate::finiteElement::boundaryConditions::Essential
essential (Dirichlet condition) for FE based problems

boundaryName (req) 
: (string) the name for this boundary condition

labelIds (req) 
: (int list) the ids on the mesh to apply the boundary condition

boundaryValue (req) 
: (ablate::mathFunctions::FieldFunction) the field function used to describe the boundary

labelName
: (string) the mesh label holding the boundary ids (default marker)

# ablate::mathFunctions::MathFunction
## ablate::mathFunctions::SimpleFormula*
a string based function to be parsed with muparser. The (string) formula that may accept x, y, z, t as variables

## ablate::mathFunctions::ConstantValue
sets a constant value to all values in field

## ablate::mathFunctions::ParsedSeries
 computes a series result from a string function with variables x, y, z, t, and i where i index of summation. $$\sum_{i = m}^n formula(x, y, z, t, n)$$

formula (req) 
: (string) see ParsedFunction for details on the string formatting.

lowerBound (req) 
: (int) the inclusive lower bound of summation (m)

upperBound (req) 
: (int) the inclusive upper bound of summation (n)

## ablate::mathFunctions::LinearTable
A table that is built from a spreadsheet that allows linear interpolation of variables based on monotonically increasing independent variables

file (req) 
: (file path or url) a file with csv data and header

independent (req) 
: (string) the name of the independent column name as defined in the header

dependent (req) 
: (string list) the names of the dependent column in the order in which to apply them

mappingFunction (req) 
: (ablate::mathFunctions::MathFunction)  the function that maps from the physical x,y,z, and t space to the table independent variable

## ablate::mathFunctions::Formula
 computes string function with variables x, y, z, and t where additional variables can be specified using other functions

formula (req) 
: (string) see ParsedFunction for details on the string formatting.

nested
: (std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, ablate::mathFunctions::MathFunction, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, ablate::mathFunctions::MathFunction> > >) a map of nested MathFunctions.  These functions are assumed to compute a single scalar value

constants
: (ablate::parameters::Parameters) constants that can be used in the formula

## ablate::mathFunctions::geom::Sphere
assigns a uniform value to all points inside the sphere

center (req) 
: (double list) the sphere center

radius
: (double) the sphere radius

insideValues
: (ablate::mathFunctions::MathFunction) the values for inside the sphere, defaults to 1

outsideValues
: (ablate::mathFunctions::MathFunction) the outside values, defaults to zero

## ablate::mathFunctions::geom::Box
assigns a uniform value to all points inside the box

lower (req) 
: (double list) the box lower corner

upper (req) 
: (double list) the box upper corner

insideValues
: (ablate::mathFunctions::MathFunction) the values for inside the sphere, defaults to 1

outsideValues
: (ablate::mathFunctions::MathFunction) the outside values, defaults to zero

## ablate::mathFunctions::geom::Surface
Assigned a unified number to all points inside of cad geometry file.

path (req) 
: (file path or url) the path to the step/stp file

insideValues
: (ablate::mathFunctions::MathFunction) the values for inside the sphere, defaults to 1

outsideValues
: (ablate::mathFunctions::MathFunction) the outside values, defaults to zero

egadsVerboseLevel
: (int) the egads verbose level for output (default is 0, max is 3)

# ablate::mathFunctions::FieldFunction
## ablate::finiteVolume::fieldFunctions::Euler
initializes the euler conserved field variables based upon a CompressibleFlowState

state (req) 
: (ablate::finiteVolume::fieldFunctions::CompressibleFlowState) The CompressibleFlowState used to initialize

region
: (ablate::domain::Region) A subset of the domain to apply the field function

## ablate::finiteVolume::fieldFunctions::MassFractions
initializes the yi field function variables based upon a the list of functions and eos

eos (req) 
: (ablate::eos::EOS) The eos with the list of species

values (req) 
: (std::vector<ablate::mathFunctions::FieldFunction, std::allocator<ablate::mathFunctions::FieldFunction> >) The list of mass fraction functions

## ablate::finiteVolume::fieldFunctions::DensityMassFractions
initializes the densityYi conserved field variables based upon a CompressibleFlowState

state (req) 
: (ablate::finiteVolume::fieldFunctions::CompressibleFlowState) The CompressibleFlowState used to initialize

region
: (ablate::domain::Region) A subset of the domain to apply the field function

## ablate::finiteVolume::fieldFunctions::DensityExtraVariables
initializes the densityEV conserved field variables based upon a CompressibleFlowState and specified EV

state (req) 
: (ablate::finiteVolume::fieldFunctions::CompressibleFlowState) The CompressibleFlowState used to initialize

functions (req) 
: (std::vector<ablate::mathFunctions::MathFunction, std::allocator<ablate::mathFunctions::MathFunction> >) The EV values in order

## ablate::mathFunctions::FieldFunction*
a field description that can be used for initialization or exact solution 

fieldName (req) 
: (string) the field name

field (req) 
: (ablate::mathFunctions::MathFunction) the math function used to describe the field

timeDerivative
: (ablate::mathFunctions::MathFunction) the math function used to describe the field time derivative

region
: (ablate::domain::Region) A subset of the domain to apply the field function

# ablate::boundarySolver::BoundaryProcess
## ablate::boundarySolver::lodi::IsothermalWall
Enforces a isothermal wall with fixed velocity/temperature

eos (req) 
: (ablate::eos::EOS) The EOS describing the flow field at the wall

pgs
: (ablate::finiteVolume::processes::PressureGradientScaling) Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows

## ablate::boundarySolver::lodi::OpenBoundary
Treats boundary as open.

eos (req) 
: (ablate::eos::EOS) The EOS describing the flow field at the boundary

reflectFactor (req) 
: (double) boundary reflection factor

referencePressure (req) 
: (double) reference pressure

maxAcousticsLength (req) 
: (double) maximum length in the domain for acoustics to propagate 

pgs
: (ablate::finiteVolume::processes::PressureGradientScaling) Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows

## ablate::boundarySolver::lodi::Inlet
Enforces an inlet with specified velocity

eos (req) 
: (ablate::eos::EOS) The EOS describing the flow field at the wall

pgs
: (ablate::finiteVolume::processes::PressureGradientScaling) Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows

velocity
: (ablate::mathFunctions::MathFunction) optional velocity function that can change over time

## ablate::boundarySolver::physics::Sublimation
Adds in the euler/yi sources for a sublimating material.  Should be used with a LODI boundary.

latentHeatOfFusion (req) 
: (double) the latent heat of fusion [J/kg]

effectiveConductivity (req) 
: (double) the effective conductivity to compute heat flux to the surface [W/(m⋅K)]

massFractions
: (ablate::mathFunctions::FieldFunction) the species to deposit the off gas mass to (required if solving species)

additionalHeatFlux
: (ablate::mathFunctions::MathFunction) additional normal heat flux into the solid function

# ablate::io::Serializer
## ablate::io::Hdf5Serializer*
default serializer for IO

interval (req) 
: (ablate::io::interval::Interval) The interval object used to determine write interval.

## ablate::io::Hdf5MultiFileSerializer
serializer for IO that writes each time to a separate hdf5 file

interval (req) 
: (ablate::io::interval::Interval) The interval object used to determine write interval.

options
: (ablate::parameters::Parameters) options for the viewer passed directly to PETSc including (hdf5ViewerView, viewer_hdf5_collective, viewer_hdf5_sp_output

# ablate::io::interval::Interval
## ablate::io::interval::FixedInterval*
Default interval that outputs every n steps

## ablate::io::interval::SimulationTimeInterval
Outputs every dt simulation seconds. This will not result in uniform output unless the simulation dt matches.

## ablate::io::interval::WallTimeInterval
Outputs approximately every n wall time seconds

# ablate::eos::EOS
## ablate::eos::PerfectGas
perfect gas eos

parameters (req) 
: (ablate::parameters::Parameters) parameters for the perfect gas eos

species
: (string list) species to track.  Note: species mass fractions do not change eos

## ablate::eos::TChem
TChem ideal gas eos

mechFile (req) 
: (file path or url) the mech file (CHEMKIN Format)

thermoFile (req) 
: (file path or url) the thermo file (CHEMKIN Format)

## ablate::eos::StiffenedGas
stiffened gas eos

parameters (req) 
: (ablate::parameters::Parameters) parameters for the stiffened gas eos

species
: (string list) species to track.  Note: species mass fractions do not change eos

# ablate::eos::transport::TransportModel
## ablate::eos::transport::Constant*
constant value transport model (often used for testing)

k
: (double) thermal conductivity [W/(m K)]

mu
: (double) viscosity [Pa s]

diff
: (double) diffusivity [m2/s]

## ablate::eos::transport::Sutherland
Sutherland Transport model

eos (req) 
: (ablate::eos::EOS) The EOS used to compute Cp (needed for Conductivity)

# ablate::domain::FieldDescriptor
## ablate::finiteVolume::CompressibleFlowFields
FVM fields need for compressible flow

eos (req) 
: (ablate::eos::EOS) the equation of state to be used for the flow

extraVariables
: (string list) Any extra variables to transport

region
: (ablate::domain::Region) the region for the compressible flow (defaults to entire domain)

conservedFieldOptions
: (ablate::parameters::Parameters) petsc options used for the conserved fields.  Common options would be petscfv_type and petsclimiter_type

## ablate::finiteElement::LowMachFlowFields
FE fields need for incompressible/low-Mach flow

region
: (ablate::domain::Region) the region for the compressible flow (defaults to entire domain)

includeSourceTerms
: (bool) include aux fields for source terms (defaults to false)

## ablate::domain::FieldDescription*
A single custom field description

name (req) 
: (string) the name of the field

prefix
: (string) optional prefix (defaults to name)

components
: (string list) the components in the field (defaults to 1)

location
: (cppParser::EnumWrapper<ablate::domain::FieldLocation>) if it is a solution (SOL) or auxiliary (aux) field

type (req) 
: (cppParser::EnumWrapper<ablate::domain::FieldType>) if it is a finite volume (FV) or finite element (FE) field

region
: (ablate::domain::Region) the region in which this field lives

options
: (ablate::parameters::Parameters) field specific options

# ablate::domain::FieldDescription
## ablate::domain::FieldDescription*
A single custom field description

name (req) 
: (string) the name of the field

prefix
: (string) optional prefix (defaults to name)

components
: (string list) the components in the field (defaults to 1)

location
: (cppParser::EnumWrapper<ablate::domain::FieldLocation>) if it is a solution (SOL) or auxiliary (aux) field

type (req) 
: (cppParser::EnumWrapper<ablate::domain::FieldType>) if it is a finite volume (FV) or finite element (FE) field

region
: (ablate::domain::Region) the region in which this field lives

options
: (ablate::parameters::Parameters) field specific options

# ablate::domain::Domain
## ablate::domain::BoxMesh
simple uniform box mesh

name (req) 
: (string) the name of the domain/mesh object

fields
: (std::vector<ablate::domain::FieldDescriptor, std::allocator<ablate::domain::FieldDescriptor> >) a list of fields/field descriptors

modifiers
: (std::vector<ablate::domain::modifiers::Modifier, std::allocator<ablate::domain::modifiers::Modifier> >) a list of domain modifier

faces (req) 
: (int list) the number of faces in each direction

lower (req) 
: (double list) the lower bound of the mesh

upper (req) 
: (double list) the upper bound of the mesh

boundary
: (string list) custom boundary types (NONE, GHOSTED, MIRROR, PERIODIC)

simplex
: (bool) sets if the elements/cells are simplex

## ablate::domain::DMPlex
DMPlex that can be set using PETSc options

fields
: (std::vector<ablate::domain::FieldDescriptor, std::allocator<ablate::domain::FieldDescriptor> >) a list of fields/field descriptors

name (req) 
: (string) the mesh dm name

modifiers
: (std::vector<ablate::domain::modifiers::Modifier, std::allocator<ablate::domain::modifiers::Modifier> >) a list of domain modifier

## ablate::domain::FileMesh
read a DMPlex from a file

name (req) 
: (string) the name of the domain/mesh object

path (req) 
: (file path or url) the path to the mesh file

fields
: (std::vector<ablate::domain::FieldDescriptor, std::allocator<ablate::domain::FieldDescriptor> >) a list of fields/field descriptors

modifiers
: (std::vector<ablate::domain::modifiers::Modifier, std::allocator<ablate::domain::modifiers::Modifier> >) a list of domain modifier

## ablate::domain::BoxMeshBoundaryCells
simple uniform box mesh with boundary solver cells.  It labels the boundary cells as boundaryCells and boundaryCellsLeft, etc

name (req) 
: (string) the name of the domain/mesh object

fields
: (std::vector<ablate::domain::FieldDescriptor, std::allocator<ablate::domain::FieldDescriptor> >) a list of fields/field descriptors

preModifiers
: (std::vector<ablate::domain::modifiers::Modifier, std::allocator<ablate::domain::modifiers::Modifier> >) a list of domain modifiers to apply before ghost labeling

postModifiers
: (std::vector<ablate::domain::modifiers::Modifier, std::allocator<ablate::domain::modifiers::Modifier> >) a list of domain modifiers to apply after ghost labeling

faces (req) 
: (int list) the number of faces in each direction

lower (req) 
: (double list) the lower bound of the mesh

upper (req) 
: (double list) the upper bound of the mesh

mainRegion (req) 
: (ablate::domain::Region) the label for the main region (no ghost cells)

boundaryFaceRegion (req) 
: (ablate::domain::Region) the label for the new face cells between regions

simplex
: (bool) sets if the elements/cells are simplex

# ablate::domain::Region
## ablate::domain::Region*
The region in which this solver applies (Label & Values)

name (req) 
: (string) the label name

value
: (int) the value on the label (default is 1)

# ablate::domain::modifiers::Modifier
## ablate::domain::modifiers::DistributeWithGhostCells
Distribute DMPlex with ghost cells

ghostCellDepth
: (int) the number of ghost cells to share on the boundary.  Default is 1.

## ablate::domain::modifiers::GhostBoundaryCells
Adds ghost cells to the boundary

labelName
: (string) The label specifying the boundary faces, or "Face Sets" if not specified

## ablate::domain::modifiers::SetFromOptions
Sets the specified options on the dm.

## ablate::domain::modifiers::CreateLabel
Creates a new label for all positive points in the function

region (req) 
: (ablate::domain::Region) the region describing the new label

function (req) 
: (ablate::mathFunctions::MathFunction) the function to evaluate

depth
: (int) The depth in which to apply the label.  The default is zero or cell/element

## ablate::domain::modifiers::TagLabelBoundary
Creates a new label for all faces on the outside of the boundary

region (req) 
: (ablate::domain::Region) the region to tag the boundary

boundaryFaceRegion (req) 
: (ablate::domain::Region) the new region for the boundary faces

boundaryCellRegion
: (ablate::domain::Region) the new region for the boundary cells

## ablate::domain::modifiers::MergeLabels
Creates a new label for all faces on the outside of the boundary

mergedRegion (req) 
: (ablate::domain::Region) the merged region to create

regions (req) 
: (std::vector<ablate::domain::Region, std::allocator<ablate::domain::Region> >) the regions to include in the new merged region

## ablate::domain::modifiers::IntersectLabels
Creates a new label that intersections the provided regions

intersectRegion (req) 
: (ablate::domain::Region) the intersect region to create/used

regions (req) 
: (std::vector<ablate::domain::Region, std::allocator<ablate::domain::Region> >) the regions to include in the intersection

## ablate::domain::modifiers::SubtractLabel
Cuts/removes the given region (difference = minuend - subtrahend)

differenceRegion (req) 
: (ablate::domain::Region) the result of the operation

minuendRegion (req) 
: (ablate::domain::Region) the minuend region

subtrahendRegion (req) 
: (ablate::domain::Region) the region to be removed

## ablate::domain::modifiers::TagLabelInterface
Class to label/tag all faces/cells on the interface between two labels.  The left/right designations are just used to separate the left/right labels.

leftRegion (req) 
: (ablate::domain::Region) the "left" region

rightRegion (req) 
: (ablate::domain::Region) the "right" region

boundaryFaceRegion (req) 
: (ablate::domain::Region) the new region for the newly tagged boundary faces

leftBoundaryCellRegion
: (ablate::domain::Region) optional new region to tag the boundary cells on the "left" of region

rightBoundaryCellRegion
: (ablate::domain::Region) optional new region to tag the boundary cells on the "right" of region

# ablate::solver::TimeStepper
## ablate::solver::TimeStepper*
the basic stepper

name (req) 
: (string) the time stepper name

domain (req) 
: (ablate::domain::Domain) the mesh used for the simulation

arguments (req) 
: (argument map) arguments to be passed to petsc

io
: (ablate::io::Serializer) the serializer used with this timestepper

initialization
: (std::vector<ablate::mathFunctions::FieldFunction, std::allocator<ablate::mathFunctions::FieldFunction> >) initialization field functions

exactSolution
: (std::vector<ablate::mathFunctions::FieldFunction, std::allocator<ablate::mathFunctions::FieldFunction> >) optional exact solutions that can be used for error calculations

absoluteTolerances
: (std::vector<ablate::mathFunctions::FieldFunction, std::allocator<ablate::mathFunctions::FieldFunction> >) optional absolute tolerances for a field

relativeTolerances
: (std::vector<ablate::mathFunctions::FieldFunction, std::allocator<ablate::mathFunctions::FieldFunction> >) optional relative tolerances for a field

# ablate::solver::Solver
## ablate::finiteVolume::FiniteVolumeSolver
finite volume solver

id (req) 
: (string) the name of the flow field

region
: (ablate::domain::Region) the region to apply this solver.  Default is entire domain

options
: (ablate::parameters::Parameters) the options passed to PETSC for the flow

processes (req) 
: (std::vector<ablate::finiteVolume::processes::Process, std::allocator<ablate::finiteVolume::processes::Process> >) the processes used to describe the flow

boundaryConditions
: (std::vector<ablate::finiteVolume::boundaryConditions::BoundaryCondition, std::allocator<ablate::finiteVolume::boundaryConditions::BoundaryCondition> >) the boundary conditions for the flow field

computePhysicsTimeStep
: (bool) determines if a physics based time step is used to control the FVM time stepping (default is false)

## ablate::finiteVolume::CompressibleFlowSolver
compressible finite volume flow

id (req) 
: (string) the name of the flow field

region
: (ablate::domain::Region) the region to apply this solver.  Default is entire domain

options
: (ablate::parameters::Parameters) the options passed to PETSc

eos (req) 
: (ablate::eos::EOS) the equation of state used to describe the flow

parameters
: (ablate::parameters::Parameters) the parameters used for field values

transport
: (ablate::eos::transport::TransportModel) the diffusion transport model

fluxCalculator
: (ablate::finiteVolume::fluxCalculator::FluxCalculator) the flux calculators (defaults to none)

additionalProcesses
: (std::vector<ablate::finiteVolume::processes::Process, std::allocator<ablate::finiteVolume::processes::Process> >) any additional processes besides euler/yi/ev transport

boundaryConditions
: (std::vector<ablate::finiteVolume::boundaryConditions::BoundaryCondition, std::allocator<ablate::finiteVolume::boundaryConditions::BoundaryCondition> >) the boundary conditions for the flow field

computePhysicsTimeStep
: (bool) determines if a physics based time step is used to control the FVM time stepping (default is false)

## ablate::finiteVolume::ReactingCompressibleFlowSolver
reacting compressible finite volume flow

id (req) 
: (string) the name of the flow field

region
: (ablate::domain::Region) the region to apply this solver.  Default is entire domain

options
: (ablate::parameters::Parameters) the options passed to PETSc

eos (req) 
: (ablate::eos::EOS) the TChem v1 equation of state used to describe the flow

parameters (req) 
: (ablate::parameters::Parameters) the compressible flow parameters cfl, gamma, etc.

transport
: (ablate::eos::transport::TransportModel) the diffusion transport model

fluxCalculator
: (ablate::finiteVolume::fluxCalculator::FluxCalculator) the flux calculator (defaults to AUSM)

boundaryConditions
: (std::vector<ablate::finiteVolume::boundaryConditions::BoundaryCondition, std::allocator<ablate::finiteVolume::boundaryConditions::BoundaryCondition> >) the boundary conditions for the flow field

additionalProcesses
: (std::vector<ablate::finiteVolume::processes::Process, std::allocator<ablate::finiteVolume::processes::Process> >) any additional processes besides euler/yi/ev transport and rxn

computePhysicsTimeStep
: (bool) determines if a physics based time step is used to control the FVM time stepping (default is false)

## ablate::finiteElement::LowMachFlowSolver
incompressible FE flow

id (req) 
: (string) the name of the flow field

region
: (ablate::domain::Region) the region to apply this solver.  Default is entire domain

options
: (ablate::parameters::Parameters) options for the flow passed directly to PETSc

parameters (req) 
: (ablate::parameters::Parameters) the flow field parameters

boundaryConditions (req) 
: (std::vector<ablate::finiteElement::boundaryConditions::BoundaryCondition, std::allocator<ablate::finiteElement::boundaryConditions::BoundaryCondition> >) the boundary conditions for the flow field

auxFields (req) 
: (std::vector<ablate::mathFunctions::FieldFunction, std::allocator<ablate::mathFunctions::FieldFunction> >) enables and sets the update functions for the auxFields

## ablate::finiteElement::IncompressibleFlowSolver
incompressible FE flow

id (req) 
: (string) the name of the flow field

region
: (ablate::domain::Region) the region to apply this solver.  Default is entire domain

options
: (ablate::parameters::Parameters) options for the flow passed directly to PETSc

parameters (req) 
: (ablate::parameters::Parameters) the flow field parameters

boundaryConditions (req) 
: (std::vector<ablate::finiteElement::boundaryConditions::BoundaryCondition, std::allocator<ablate::finiteElement::boundaryConditions::BoundaryCondition> >) the boundary conditions for the flow field

auxFields
: (std::vector<ablate::mathFunctions::FieldFunction, std::allocator<ablate::mathFunctions::FieldFunction> >) enables and sets the update functions for the auxFields

## ablate::boundarySolver::BoundarySolver
A solver used to compute boundary values in boundary cells

id (req) 
: (string) the name of the flow field

region (req) 
: (ablate::domain::Region) the region to apply this solver.

fieldBoundary (req) 
: (ablate::domain::Region) the region describing the faces between the boundary and field

processes (req) 
: (std::vector<ablate::boundarySolver::BoundaryProcess, std::allocator<ablate::boundarySolver::BoundaryProcess> >) a list of boundary processes

options
: (ablate::parameters::Parameters) the options passed to PETSC for the flow

## ablate::particles::Tracer
massless particles that advect with the flow

id (req) 
: (string) the name of this particle solver

region
: (ablate::domain::Region) the region to apply this solver.  Default is entire domain

options
: (ablate::parameters::Parameters) options for the flow passed directly to PETSc

ndims (req) 
: (int) the number of dimensions for the particle

initializer (req) 
: (ablate::particles::initializers::Initializer) the initial particle setup methods

exactSolution
: (ablate::mathFunctions::MathFunction) the particle location exact solution

## ablate::particles::Inertial
particles (with mass) that advect with the flow

id (req) 
: (string) the name of this particle solver

region
: (ablate::domain::Region) the region to apply this solver.  Default is entire domain

options
: (ablate::parameters::Parameters) options for the flow passed directly to PETSc

ndims (req) 
: (int) the number of dimensions for the particle

parameters (req) 
: (ablate::parameters::Parameters) fluid parameters for the particles (fluidDensity, fluidViscosity, gravityField)

initializer (req) 
: (ablate::particles::initializers::Initializer) the initial particle setup methods

fieldInitialization (req) 
: (std::vector<ablate::mathFunctions::FieldFunction, std::allocator<ablate::mathFunctions::FieldFunction> >) the initial particle fields setup methods

exactSolution
: (ablate::mathFunctions::MathFunction) the particle location/velocity exact solution

# ablate::monitors::logs::Log
## ablate::monitors::logs::CsvLog
Writes the result of the log to a csv file.  Only prints data to the log.

name (req) 
: (string) the name of the log file

## ablate::monitors::logs::StreamLog
Writes to the std::cout stream

## ablate::monitors::logs::FileLog
Writes the result of the log to a file

name (req) 
: (string) the name of the log file

## ablate::monitors::logs::StdOut
Writes to the standard out

# ablate::monitors::probes::ProbeInitializer
## ablate::monitors::probes::List*
A simple list of probes

## ablate::monitors::probes::Rake
Inserts probes along a line

name (req) 
: (string) The rake name

start (req) 
: (double list) the starting point for the rake

end (req) 
: (double list) the ending point for the rake

number (req) 
: (int) the number of probes in the rake

# ablate::monitors::probes::Probe
## ablate::monitors::probes::Probe*
Probe specification struct

name (req) 
: (string) name of the probe

location (req) 
: (double list) the probe location

# ablate::monitors::Monitor
## ablate::monitors::FieldErrorMonitor
Computes and reports the error every time step

log
: (ablate::monitors::logs::Log) where to record log (default is stdout)

## ablate::monitors::SolutionErrorMonitor
Computes and reports the error every time step

scope (req) 
: (cppParser::EnumWrapper<ablate::monitors::SolutionErrorMonitor::Scope>) how the error should be calculated ('vector', 'component')

type (req) 
: (cppParser::EnumWrapper<ablate::monitors::SolutionErrorMonitor::Norm>) norm type ('l1','l1_norm','l2', 'linf', 'l2_norm')

log
: (ablate::monitors::logs::Log) where to record log (default is stdout)

## ablate::monitors::TimeStepMonitor
Reports the current step, time, and dt

log
: (ablate::monitors::logs::Log) where to record log (default is stdout)

interval
: (ablate::io::interval::Interval) report interval object, defaults to every

## ablate::monitors::IgnitionDelayPeakYi
Compute the ignition time based upon peak mass fraction

species (req) 
: (string) the species used to determine the peak Yi

location (req) 
: (double list) the monitor location

log
: (ablate::monitors::logs::Log) where to record the final ignition time (default is stdout)

historyLog
: (ablate::monitors::logs::Log) where to record the time and yi history (default is none)

## ablate::monitors::IgnitionDelayTemperature
Compute the ignition time based upon temperature change

eos (req) 
: (ablate::eos::EOS) the eos used to compute temperature

location (req) 
: (double list) the monitor location

thresholdTemperature (req) 
: (double) the temperature used to define ignition delay

log
: (ablate::monitors::logs::Log) where to record the final ignition time (default is stdout)

historyLog
: (ablate::monitors::logs::Log) where to record the time and yi history (default is none)

## ablate::monitors::ExtractLineMonitor
Outputs the results along a line as a curve file (beta)

interval (req) 
: (int) output interval

prefix (req) 
: (string) the file prefix

start (req) 
: (double list) the line start location

end (req) 
: (double list) the line end location

outputFields (req) 
: (string list) a list of fields to write to the curve

outputAuxFields (req) 
: (string list) a list of aux fields to write to the curve 

## ablate::monitors::DmViewFromOptions
replicates the DMViewFromOptions function in PETSC

scope (req) 
: (cppParser::EnumWrapper<ablate::monitors::DmViewFromOptions::Scope>) determines if DMViewFromOptions is called initially (initial) or every time step (monitor)

options
: (string) if provided these options are used for the DMView call, otherwise global options is used

optionName
: (string) if provided the optionsName is used for DMViewFromOptions.  Needed if using global options.

## ablate::monitors::ParticleCount
Outputs the total number of particles in the domain

interval
: (int) output interval

log
: (ablate::monitors::logs::Log) where to record log (default is stdout)

## ablate::monitors::ParticleAverage
Outputs the average particle location in the domain

interval
: (int) output interval

log
: (ablate::monitors::logs::Log) where to record log (default is stdout)

## ablate::monitors::CurveMonitor
Write 1D results to a curve file

interval
: (ablate::io::interval::Interval) output interval

prefix
: (string) the file prefix

## ablate::monitors::MaxMinAverage
Prints the min/max/average for a field

field (req) 
: (string) the name of the field

log
: (ablate::monitors::logs::Log) where to record log (default is stdout)

interval
: (ablate::io::interval::Interval) report interval object, defaults to every

## ablate::monitors::PhysicsTimeStep
Reports the physics based time stepping from the FVM without enforcing it

log
: (ablate::monitors::logs::Log) where to record log (default is stdout)

interval
: (ablate::io::interval::Interval) report interval object, defaults to every

## ablate::monitors::Probes
Records the values of the specified variables at a specific point in space

probes (req) 
: (ablate::monitors::probes::ProbeInitializer) where to record log (default is stdout)

variables (req) 
: (string list) list of variables to output

interval
: (ablate::io::interval::Interval) report interval object, defaults to every

bufferSize
: (int) how often the probe file is written (default is 100, must be > 0)

# ablate::chemistry::ChemistryModel
## ablate::chemistry::ChemTabModel
Uses a tensorflow model developed by ChemTab

path (req) 
: (file path or url) the path to the model

# ablate::particles::initializers::Initializer
## ablate::particles::initializers::CellInitializer
simple cell initializer that puts particles in every element

particlesPerCellPerDim (req) 
: (int) particles per cell per dimension

## ablate::particles::initializers::BoxInitializer
simple box initializer that puts particles in a defined box

lower (req) 
: (double list) the lower bound of the box

upper (req) 
: (double list) the upper bound of the box

particlesPerDim (req) 
: (int) the particles per box dimension

# ablate::particles::drag::DragModel
## ablate::particles::drag::Linear
Computes drag according to Stokes' law.

## ablate::particles::drag::Quadratic
Computes drag according to a high Reynolds number drag model for solid spheres.


