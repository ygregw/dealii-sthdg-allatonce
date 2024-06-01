#include <deal.II/base/config.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{ using namespace dealii::LinearAlgebraPETSc; }

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/reference_cell.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <iostream>
#include <sys/resource.h>
#include <unistd.h>
#include "sthdg_error_estimator.h"
#include "sthdg_error_estimator.templates.h"

//#define SemiUpwinding
#define SemiCentreFlux

// choose a problem
#define ROTATINGPULSE
//#define INTERIORLAYER
//#define BOUNDARYLAYER

long get_mem_usage(){
	struct rusage myusage;
	getrusage(RUSAGE_SELF, &myusage);
	return myusage.ru_maxrss;
}

namespace SpaceTimeAdvecDiffuIPH
{
	using namespace dealii;

	template <int dim>
		class Solution : public Function<dim> {
			public:
				Solution (const double nu) : Function<dim>(), nu(nu) {}
				double value (const Point<dim> &p, const unsigned int component = 0) const override;
				Tensor<1, dim> gradient (const Point<dim> &p,
						const unsigned int /*component*/ = 0) const override;
			private:
				const double nu;
		};

	template <int dim>
		double Solution<dim>::value (const Point<dim> &p, const unsigned int) const {
			double return_value = 0;
			const double t = p[0];
			const double x = p[1];
			const double y = p[2];
			switch (dim)
			{
				case 3:
					{
#ifdef ROTATINGPULSE
						const double x1 = x*std::cos(4*t)+y*std::sin(4*t);
						const double x2 = -x*std::sin(4*t)+y*std::cos(4*t);
						const double x1c = -0.2;
						const double x2c = 0.1;
						const double sigma_sq = 0.01; // sigma * sigma
						const double x1d_sq = (x1-x1c)*(x1-x1c);
						const double x2d_sq = (x2-x2c)*(x2-x2c);
						return_value =
							std::exp(-(x1d_sq+x2d_sq)/(2*sigma_sq+4*nu*t))
							* sigma_sq/(sigma_sq+2*nu*t);
#elif defined(INTERIORLAYER)
						// the manufactured solution has the form u(t,x,y) = k(t)*q(x,y);
						const double nnu = nu; // new nu
						const double k = (1-std::exp(-t));
						const double q_atan = std::atan((y-x)/(nnu*std::sqrt(2)));
						const double q = q_atan * (1-(x+y)*(x+y)/2);
						return_value = k*q;
#elif defined(BOUNDARYLAYER)
						// the manufactured solution has the form u(t,x,y) = k(t)*q(x)*r(y);
						const double nnu = nu; // new nu
						const double k = (1-std::exp(-t));
						const double q = (std::exp((x-1)/nnu)-1)/(std::exp(-1/nnu)-1)+x-1;
						const double r = (std::exp((y-1)/nnu)-1)/(std::exp(-1/nnu)-1)+y-1;
						return_value = k*q*r;
#endif
						break;
					}
				default:
					{
						Assert (false, ExcNotImplemented());
					}
			}
			return return_value;
		};

	template <int dim>
		Tensor<1,dim> Solution<dim>::gradient (const Point<dim> &p, const unsigned int) const {
			Tensor<1,dim> return_value;
			const double t = p[0];
			const double x = p[1];
			const double y = p[2];
			switch (dim)
			{
				case 3:
					{
#ifdef ROTATINGPULSE
						const double x1 = x*std::cos(4*t)+y*std::sin(4*t);
						const double x2 = -x*std::sin(4*t)+y*std::cos(4*t);
						const double x1c = -0.2;
						const double x2c = 0.1;
						const double x1d = x1-x1c;
						const double x2d = x2-x2c;
						const double xd_sq = x1d*x1d + x2d*x2d;
						const double sigma_sq = 0.01; // sigma * sigma
						const double sigma_nut = sigma_sq + 2*nu*t;
						const double exp_term = std::exp(-xd_sq/(2*sigma_nut));
						return_value[0] =
							sigma_sq/(sigma_nut*sigma_nut) * exp_term *
							( -2*nu + nu*xd_sq/(sigma_nut) - 4*(x1d*x2-x2d*x1));
						return_value[1] =
							- sigma_sq/(sigma_nut*sigma_nut) * exp_term *
							(x1d*std::cos(4*t)-x2d*std::sin(4*t));
						return_value[2] =
							- sigma_sq/(sigma_nut*sigma_nut) * exp_term *
							(x1d*std::sin(4*t)+x2d*std::cos(4*t));
#elif defined(INTERIORLAYER)
						// the manufactured solution has the form u(t,x,y) = k(t)*q(x,y);
						// example: qx stands for partial derivative w.r.t. x of q(x,y).
						const double nnu = nu; // new nu
						const double k = (1-std::exp(-t));
						const double kt = std::exp(-t);
						const double q_atan = std::atan((y-x)/(nnu*std::sqrt(2)));
						const double q = q_atan * (1-(x+y)*(x+y)/2);
						const double q_partial_comp =
							nnu*((x+y)*(x+y)-2)/(std::sqrt(2)*(2*nnu*nnu+(y-x)*(y-x)));
						const double qx = q_partial_comp -(x+y)*q_atan;
						const double qy = -q_partial_comp -(x+y)*q_atan;
						return_value[0] = kt*q;
						return_value[1] = k*qx;
						return_value[2] = k*qy;
#elif defined(BOUNDARYLAYER)
						// the manufactured solution has the form u(t,x,y) = k(t)*q(x)*r(y);
						const double nnu = nu; // new nu
						const double k = (1-std::exp(-t));
						const double kt = std::exp(-t);
						const double q = (std::exp((x-1)/nnu)-1)/(std::exp(-1/nnu)-1)+x-1;
						const double qx = (1/nnu)*(std::exp((x-1)/nnu))/(std::exp(-1/nnu)-1)+1;
						const double r = (std::exp((y-1)/nnu)-1)/(std::exp(-1/nnu)-1)+y-1;
						const double ry = (1/nnu)*(std::exp((y-1)/nnu))/(std::exp(-1/nnu)-1)+1;
						return_value[0] = kt*q*r;
						return_value[1] = k*qx*r;
						return_value[2] = k*q*ry;
#endif
						break;
					}
				default:
					{
						Assert (false, ExcNotImplemented());
					}
			}
			return return_value;
		};

	template <int dim>
		class AdvectionVelocity : public TensorFunction<1,dim> {
			public:
				AdvectionVelocity() : TensorFunction<1,dim>() {}
				Tensor<1,dim> value (const Point<dim> &p) const override;
		};

	template <int dim>
		Tensor<1,dim> AdvectionVelocity<dim>::value(const Point<dim> &p) const {
			Tensor<1,dim> advection;
			switch (dim)
			{
				case 3:
					{
#ifdef ROTATINGPULSE
						const double xx = p[1];
						const double yy = p[2];
						advection[0] = 1;
						advection[1] = -4*yy;
						advection[2] = 4*xx;
#elif defined(INTERIORLAYER)
						(void)p;
						advection[0] = 1;
						advection[1] = 1;
						advection[2] = 1;
#elif defined(BOUNDARYLAYER)
						(void)p;
						advection[0] = 1;
						advection[1] = 1;
						advection[2] = 1;
#endif
						break;
					}
				default:
					{
						Assert(false, ExcNotImplemented());
					}
			}
			return advection;
		};

	template <int dim>
		class RightHandSide : public Function<dim> {
		public:
			RightHandSide (const double nu) : Function<dim>(), nu(nu) {}
			double value (const Point<dim> &p, const unsigned int component = 0) const override;
		private:
			const AdvectionVelocity<dim> advection_velocity;
			const double nu;
		};

	template <int dim>
		double RightHandSide<dim>::value (const Point<dim> &p, const unsigned int) const {
			double return_value = 0;
			switch (dim)
			{
				case 3:
					{
#ifdef ROTATINGPULSE
						(void)p;
						return_value = 0;
#elif defined(INTERIORLAYER)
						const double t = p[0];
						const double x = p[1];
						const double y = p[2];
						// the manufactured solution has the form u(t,x,y) = k(t)*q(x,y);
						// example: qx stands for partial derivative w.r.t. x of q(x,y).
						const double nnu = nu; // new nu
						const double k = (1-std::exp(-t));
						const double kt = std::exp(-t);
						const double q_laplace_comp = 1-0.5*(x+y)*(x+y);
						const double q_partial_comp = 2*nnu*nnu+(y-x)*(y-x);
						const double q_atan = std::atan((y-x)/(std::sqrt(2)*nnu));
						const double q = q_atan*q_laplace_comp;
						const double qx = nnu*((x+y)*(x+y)-2)/(std::sqrt(2)*q_partial_comp) - (x+y)*q_atan;
						const double qy = -nnu*((x+y)*(x+y)-2)/(std::sqrt(2)*q_partial_comp) - (x+y)*q_atan;
						const double qxx = 2*std::sqrt(2)*nnu*(x+y)/q_partial_comp
							-4*nnu*(y-x)*q_laplace_comp/(std::sqrt(2)*q_partial_comp*q_partial_comp)
							-q_atan;
						const double qyy = -2*std::sqrt(2)*nnu*(x+y)/q_partial_comp
							-4*nnu*(y-x)*q_laplace_comp/(std::sqrt(2)*q_partial_comp*q_partial_comp)
							-q_atan;
						const double ut = kt*q;
						const double ux = k*qx;
						const double uy = k*qy;
						const double uxx = k*qxx;
						const double uyy = k*qyy;
						return_value = -nu*uxx-nu*uyy+ut+ux+uy;
#elif defined(BOUNDARYLAYER)
						const double nnu = nu; // new nu
						const double t = p[0];
						const double x = p[1];
						const double y = p[2];
						const double k = (1-std::exp(-t));
						const double kt = std::exp(-t);
						const double q = (std::exp((x-1)/nnu)-1)/(std::exp(-1/nnu)-1)+x-1;
						const double qx = (1/nnu)*(std::exp((x-1)/nnu))/(std::exp(-1/nnu)-1)+1;
						const double qxx = (qx-1)/nnu;
						const double r = (std::exp((y-1)/nnu)-1)/(std::exp(-1/nnu)-1)+y-1;
						const double ry = (1/nnu)*(std::exp((y-1)/nnu))/(std::exp(-1/nnu)-1)+1;
						const double ryy = (ry-1)/nnu;
						const double ut = kt*q*r;
						const double ux = k*qx*r;
						const double uy = k*q*ry;
						const double uxx = k*qxx*r;
						const double uyy = k*q*ryy;
						return_value = -nu*uxx-nu*uyy+ut+ux+uy;
#endif
						break;
					}
				default:
					{
						Assert (false, ExcNotImplemented());
					}
			}
			return return_value;

		};


	template <int dim>
		class SpaceTimeHDG
		{
			public:
				enum RefinementMode
				{
					uniform_refinement,
					adaptive_refinement
				};

				SpaceTimeHDG(const unsigned int degree, const double nu, const RefinementMode refinement_mode, const unsigned int num_cycle, const bool ifoutput);
				void run();

			private:
				void init_coarse_mesh(const double dt, const double T);
				void refine_mesh(const double T);
				void setup_system();
				void assemble_system(const bool reconstruct_trace = false);
				void solve();
				void calculate_errors();
				void output_results(const unsigned int cycle, const double nu);

				MPI_Comm mpi_communicator;

				parallel::distributed::Triangulation<dim> triangulation;

				// local cell solution
				FE_DGQ<dim>			fe_cell;
				DoFHandler<dim> dof_handler_cell;
				LA::MPI::Vector local_owned_sol_cell;
				LA::MPI::Vector local_relevant_sol_cell;
				IndexSet local_owned_dofs_cell;
				IndexSet local_relevant_dofs_cell;

				// global facet solution
				FE_FaceQ<dim>   fe_facet;
				DoFHandler<dim> dof_handler_facet;
				LA::MPI::Vector local_relevant_sol_facet;
				IndexSet local_owned_dofs_facet;
				IndexSet local_relevant_dofs_facet;

				AffineConstraints<double> constraints;

				LA::MPI::SparseMatrix system_matrix;
				LA::MPI::Vector       system_rhs;

				ConvergenceTable convergence_table;

				const double nu; // diffusion parameter
				const RefinementMode refinement_mode;
				const unsigned int num_cycle; // num of refinement levels

				// l2, sh1, th1, adv_facet, dif_facet, neumann, nonjump_total, total
				std::vector<double> error_list;
				Vector<double> estimated_error_per_cell;
				double estimated_error;
				double efficiency_index;

				ConditionalOStream pcout;
				TimerOutput computing_timer;

				const bool ifoutput; 
		};


	template <int dim>
		SpaceTimeHDG<dim>::SpaceTimeHDG(const unsigned int degree,
				const double nu,
				const RefinementMode refinement_mode,
				const unsigned int num_cycle,
				const bool ifoutput
				) :
			mpi_communicator(MPI_COMM_WORLD),
			triangulation(mpi_communicator),
			fe_cell(degree),
			dof_handler_cell(triangulation),
			fe_facet(degree),
			dof_handler_facet(triangulation),
			nu(nu),
			refinement_mode(refinement_mode),
			num_cycle(num_cycle),
			error_list(8, 0.0),
			estimated_error_per_cell(0),
			estimated_error(0.0),
			efficiency_index(0.0),
			pcout(std::cout,
					(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
			computing_timer(mpi_communicator,
					pcout,
					TimerOutput::never,
					TimerOutput::wall_times),
			ifoutput(ifoutput)
	{}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // initialize coarse mesh
  // 1. generate global mesh
	// 2. mark boundary face ids.
	template <int dim>
		void SpaceTimeHDG<dim>::init_coarse_mesh(const double dt, const double T)
		{
			TimerOutput::Scope t(computing_timer, "init_mesh");
			triangulation.clear();
			const double L = 0.5;
#ifdef ROTATINGPULSE
			const Point<dim,double> p1(0, -L, -L);
			const Point<dim,double> p2(T, L, L);
			unsigned int num_cell_s = floor(2*L/dt);
			unsigned int num_cell_t = floor(T/dt);
#elif defined(BOUNDARYLAYER)
			const Point<dim,double> p1(0, 0, 0);
			const Point<dim,double> p2(T, 2*L, 2*L);
			unsigned int num_cell_s = floor(2*L/dt);
			unsigned int num_cell_t = floor(T/dt);
#elif defined(INTERIORLAYER)
			const Point<dim,double> p1(0, -L, -L);
			const Point<dim,double> p2(T, L, L);
			unsigned int num_cell_s = floor(2*L/dt);
			unsigned int num_cell_t = floor(T/dt);
#endif
			const std::vector<unsigned int> repetitions = {num_cell_t, num_cell_s, num_cell_s};
			GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

			for (auto &face : triangulation.active_face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					// bottom R-boundary has id 1;
					if (std::fabs(face_center[0]) < 1e-12) {
						face->set_boundary_id(1);
					}
					// top R-boundary has id 2;
					else if (std::fabs(face_center[0]-T) < 1e-12) {
						face->set_boundary_id(2);
					} else {
						// Q-boundary has id 0;
						face->set_boundary_id(0);
					}
				}

		}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // refine mesh
	template <int dim>
		void SpaceTimeHDG<dim>::refine_mesh(const double T)
		{
			TimerOutput::Scope t(computing_timer, "refine_mesh");

			switch (refinement_mode) {
				case uniform_refinement:
					{
						triangulation.refine_global();
						break;
					}

				case adaptive_refinement:
					{
						parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation,
								estimated_error_per_cell, 0.25, 0.1);
						triangulation.execute_coarsening_and_refinement();
						break;
					}
			}

			for (auto &face : triangulation.active_face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					// bottom R-boundary has id 1;
					if (std::fabs(face_center[0]) < 1e-12) {
						face->set_boundary_id(1);
					}
					// top R-boundary has id 2;
					else if (std::fabs(face_center[0]-T) < 1e-12) {
						face->set_boundary_id(2);
					} else {
						// Q-boundary has id 0;
						face->set_boundary_id(0);
					}
				}


		}

	template <int dim>
		void SpaceTimeHDG<dim>::setup_system()
		{
			TimerOutput::Scope t(computing_timer, "setup_system");

			dof_handler_cell.distribute_dofs(fe_cell);
			local_owned_dofs_cell = dof_handler_cell.locally_owned_dofs();
			local_relevant_dofs_cell =
				DoFTools::extract_locally_relevant_dofs(dof_handler_cell);

			dof_handler_facet.distribute_dofs(fe_facet);
			local_owned_dofs_facet = dof_handler_facet.locally_owned_dofs();
			local_relevant_dofs_facet =
				DoFTools::extract_locally_relevant_dofs(dof_handler_facet);

			local_owned_sol_cell.reinit(local_owned_dofs_cell,
					mpi_communicator);
			local_relevant_sol_cell.reinit(local_owned_dofs_cell,
					local_relevant_dofs_cell,
					mpi_communicator);
			local_relevant_sol_facet.reinit(local_owned_dofs_facet,
					local_relevant_dofs_facet,
					mpi_communicator);

			system_rhs.reinit(local_owned_dofs_facet, mpi_communicator);

			// Assign Dirichlet boundary values from the pre-defined Solution class
			// Dirichlet boundary: Q-boundary
			// Neumann boundary: bottom R-boundary (simply u = g)
			constraints.clear();
			constraints.reinit(local_relevant_dofs_facet);
			DoFTools::make_hanging_node_constraints(dof_handler_facet, constraints);
			std::map<types::boundary_id, const Function<dim> *> boundary_functions;
			Solution<dim> solution_function(nu);
			boundary_functions[0] = &solution_function;
			VectorTools::interpolate_boundary_values(dof_handler_facet,
					boundary_functions,
					constraints);
			constraints.close();

			DynamicSparsityPattern dsp(local_relevant_dofs_facet);
			DoFTools::make_sparsity_pattern(dof_handler_facet, dsp, constraints, false);
			SparsityTools::distribute_sparsity_pattern(dsp,
					dof_handler_facet.locally_owned_dofs(),
					mpi_communicator,
					local_relevant_dofs_facet);
			system_matrix.reinit(local_owned_dofs_facet,
					local_owned_dofs_facet,
					dsp,
					mpi_communicator);
		}

	template <int dim>
		void SpaceTimeHDG<dim>::assemble_system(const bool trace_reconstruct)
		{
			TimerOutput::Scope t(computing_timer, "assemble_system");
			const QGauss<dim> quadrature_formula(fe_facet.degree+1);
			const QGauss<dim-1> face_quadrature_formula(fe_facet.degree+1);

			const UpdateFlags local_flags(update_values |
					update_gradients |
					update_JxW_values |
					update_quadrature_points);

			const UpdateFlags local_face_flags(update_values |
					update_gradients |
					update_JxW_values);

			const UpdateFlags flags(update_values |
					update_normal_vectors |
					update_quadrature_points |
					update_JxW_values);

			FullMatrix<double> cell_matrix(fe_facet.dofs_per_cell, fe_facet.dofs_per_cell);
			Vector<double> cell_vector(fe_facet.dofs_per_cell);
			std::vector<types::global_dof_index> local_dof_indices(fe_facet.dofs_per_cell);

			FEValues<dim> fe_values_local(fe_cell,quadrature_formula,local_flags);
			FEFaceValues<dim> fe_face_values_local(fe_cell,face_quadrature_formula,local_face_flags);
			FEFaceValues<dim> fe_face_values(fe_facet,face_quadrature_formula,flags);

#if defined(SemiUpwinding) || defined(SemiCentreFlux)
			const QGauss<dim-1> face_quadrature_formula_betamax(2);
			const UpdateFlags flags_betamax(update_normal_vectors | update_quadrature_points);
			FEFaceValues<dim> fe_face_values_betamax(fe_facet,face_quadrature_formula_betamax,flags_betamax);
			const unsigned int n_face_q_points_betamax =
				fe_face_values_betamax.get_quadrature().size();
#endif

			AdvectionVelocity<dim> advection_velocity;
			RightHandSide<dim> right_hand_side(nu);
			const Solution<dim> exact_sol(nu);

			const unsigned int n_q_points =
				fe_values_local.get_quadrature().size();
			const unsigned int n_face_q_points =
				fe_face_values_local.get_quadrature().size();
			const unsigned int loc_dofs_per_cell =
				fe_values_local.get_fe().n_dofs_per_cell();

			FullMatrix<double> ll_matrix(fe_cell.dofs_per_cell,fe_cell.dofs_per_cell);
			FullMatrix<double> lf_matrix(fe_cell.dofs_per_cell,fe_facet.dofs_per_cell);
			FullMatrix<double> fl_matrix(fe_facet.dofs_per_cell,fe_cell.dofs_per_cell);
			FullMatrix<double> tmp_matrix(fe_facet.dofs_per_cell,fe_cell.dofs_per_cell);
			Vector<double>     l_rhs(fe_cell.dofs_per_cell);
			Vector<double>     tmp_rhs(fe_cell.dofs_per_cell);

			std::vector<double>						u_phi(fe_cell.dofs_per_cell);
			std::vector<Tensor<1, dim>>		u_phi_grad(fe_cell.dofs_per_cell);
			std::vector<Tensor<1, dim-1>> u_phi_grad_s(fe_cell.dofs_per_cell);
			std::vector<double>						tr_phi(fe_facet.dofs_per_cell);
			std::vector<double>						trace_values(face_quadrature_formula.size());
			std::vector<double>						earlier_face_values(face_quadrature_formula.size());

			std::vector<std::vector<unsigned int>> fe_cell_support_on_face(GeometryInfo<dim>::faces_per_cell);
			std::vector<std::vector<unsigned int>> fe_support_on_face(GeometryInfo<dim>::faces_per_cell);
			{
				for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
					for (unsigned int i=0; i<fe_facet.dofs_per_cell; ++i)
					{
						if (fe_facet.has_support_on_face(i,face))
							fe_support_on_face[face].push_back(i);
					}
			}

			typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_facet.begin_active(), endc = dof_handler_facet.end();
			for(; cell!=endc; ++cell)
				if (cell->is_locally_owned())
				{
					typename DoFHandler<dim>::active_cell_iterator loc_cell (&triangulation, cell->level(), cell->index(), &dof_handler_cell);

					ll_matrix = 0;
					l_rhs = 0;
					if (!trace_reconstruct)
					{
						lf_matrix = 0;
						fl_matrix = 0;
						cell_matrix = 0;
						cell_vector = 0;
					}
					fe_values_local.reinit(loc_cell);

					for (unsigned int q = 0; q < n_q_points; ++q)
					{
						const double rhs_value = right_hand_side.value(fe_values_local.quadrature_point(q));
						const Tensor<1, dim> advection = advection_velocity.value(fe_values_local.quadrature_point(q));
						const double JxW = fe_values_local.JxW(q);
						for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
						{
							u_phi[k] = fe_values_local.shape_value(k, q);
							u_phi_grad[k] = fe_values_local.shape_grad(k, q);
							for (unsigned int m = 0; m < dim-1; ++m)
								u_phi_grad_s[k][m] = u_phi_grad[k][m+1];
						}
						for (unsigned int i = 0; i < loc_dofs_per_cell; ++i)
						{
							for (unsigned int j = 0; j < loc_dofs_per_cell; ++j)
								ll_matrix(i, j) += (
										nu * u_phi_grad_s[i] * u_phi_grad_s[j] -
										(u_phi_grad[i] * advection) * u_phi[j]
										) * JxW;
							l_rhs(i) += u_phi[i] * rhs_value * JxW;
						}
					}

					for (const auto face_no : cell->face_indices())
					{
						fe_face_values_local.reinit(loc_cell, face_no);
						fe_face_values.reinit(cell, face_no);
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
						fe_face_values_betamax.reinit(cell, face_no);
						std::vector<double>	beta_flux(n_face_q_points_betamax);
						for (unsigned int q = 0; q < n_face_q_points_betamax; ++q) {
							const Point<dim> quadrature_point_betamax = fe_face_values_betamax.quadrature_point(q);
							const Tensor<1, dim> normal_betamax= fe_face_values_betamax.normal_vector(q);
							const Tensor<1, dim> advection_betamax = advection_velocity.value(quadrature_point_betamax);
							beta_flux[q] = normal_betamax * advection_betamax;
						}
						double beta_s = 0;
#if defined(SemiUpwinding)
						double beta_flux_max = beta_flux[0];
						for (unsigned int i = 1; i < beta_flux.size(); ++i) {
							if (beta_flux[i] > beta_flux_max)
								beta_flux_max = beta_flux[i];
						}
						beta_s = std::max(beta_flux_max, 0);
#else
						double abs_beta_flux_max = std::fabs(beta_flux[0]);
						for (unsigned int i = 1; i < beta_flux.size(); ++i) {
							if (std::fabs(beta_flux[i]) > abs_beta_flux_max)
								abs_beta_flux_max = std::fabs(beta_flux[i]);
						}
						beta_s = abs_beta_flux_max;
#endif
#endif
						if (trace_reconstruct)
							fe_face_values.get_function_values(local_relevant_sol_facet, trace_values);

						for (unsigned int q = 0; q < n_face_q_points; ++q)
						{
							const double JxW = fe_face_values.JxW(q);
							const Point<dim> quadrature_point = fe_face_values.quadrature_point(q);
							const Tensor<1, dim> normal = fe_face_values.normal_vector(q);
							Tensor<1, dim> normal_s = normal;
							normal_s[0]=0;
							const Tensor<1, dim> advection = advection_velocity.value(quadrature_point);

							const double hK = cell->measure() / cell->face(face_no)->measure();
							const double alpha = 8.0 * fe_facet.degree * fe_facet.degree;
							const double ip_stab = nu*alpha/hK;
							const double beta_n = advection * normal;
							const double beta_n_abs = std::abs(beta_n);

							for (unsigned int k = 0; k < fe_cell.dofs_per_cell; ++k)
							{
								u_phi[k] = fe_face_values_local.shape_value(k, q);
								u_phi_grad[k] = fe_face_values_local.shape_grad(k,q);
							}

							// global system D-CA^{-1}B\G-CA^{-1}F
							if (!trace_reconstruct)
							{
								for (unsigned int k = 0; k < fe_support_on_face[face_no].size(); ++k)
									tr_phi[k] = fe_face_values.shape_value(fe_support_on_face[face_no][k], q);

								for (unsigned int i = 0; i < fe_cell.dofs_per_cell; ++i)
									for (unsigned int j = 0; j < fe_support_on_face[face_no].size(); ++j)
									{
										const unsigned int jj = fe_support_on_face[face_no][j];
										// if R-faces else Q-faces
										if ((std::fabs(normal[0]-(1)) < 1e-12) || (std::fabs(normal[0]+(1)) < 1e-12)) {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
											lf_matrix(i, jj) +=
												(beta_n - beta_s) * u_phi[i] * tr_phi[j] * JxW;
											fl_matrix(jj, i) -=
												- beta_s * u_phi[i] * tr_phi[j] * JxW;
#else
											lf_matrix(i, jj) +=
												0.5*(beta_n - beta_n_abs) * u_phi[i] * tr_phi[j] * JxW;
											fl_matrix(jj, i) -=
												-0.5*(beta_n + beta_n_abs) * u_phi[i] * tr_phi[j] * JxW;
#endif
										}
										else {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
											lf_matrix(i, jj) +=
												((nu * u_phi_grad[i] * normal_s +
													((beta_n - beta_s) - ip_stab) * u_phi[i]
												 ) * tr_phi[j]) * JxW;

											fl_matrix(jj, i) -=
												(( nu * u_phi_grad[i] * normal_s -
													 (beta_s + ip_stab) * u_phi[i]
												 ) * tr_phi[j]) * JxW;
#else
											lf_matrix(i, jj) +=
												((nu * u_phi_grad[i] * normal_s +
													(0.5*(beta_n - beta_n_abs) - ip_stab) * u_phi[i]
												 ) * tr_phi[j]) * JxW;

											fl_matrix(jj, i) -=
												(( nu * u_phi_grad[i] * normal_s -
													 (0.5*(beta_n + beta_n_abs) + ip_stab) * u_phi[i]
												 ) * tr_phi[j]) * JxW;
#endif
										}
									}

								for (unsigned int i = 0; i < fe_support_on_face[face_no].size(); ++i)
									for (unsigned int j = 0; j < fe_support_on_face[face_no].size(); ++j)
									{
										const unsigned int ii = fe_support_on_face[face_no][i];
										const unsigned int jj = fe_support_on_face[face_no][j];
										// if R-faces else Q-faces
										if ((std::fabs(normal[0]-(1)) < 1e-12) || (std::fabs(normal[0]+(1)) < 1e-12)) {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
											cell_matrix(ii, jj) +=
												- (beta_n - beta_s) * tr_phi[i] * tr_phi[j] * JxW;
#else
											cell_matrix(ii, jj) +=
												- 0.5*(beta_n - beta_n_abs) * tr_phi[i] * tr_phi[j] * JxW;
#endif
										}
										else {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
											cell_matrix(ii, jj) +=
												( (-(beta_n - beta_s) + ip_stab)
													* tr_phi[i] * tr_phi[j]) * JxW;
#else
											cell_matrix(ii, jj) +=
												( (-0.5*(beta_n - beta_n_abs) + ip_stab)
													* tr_phi[i] * tr_phi[j]) * JxW;
#endif
										}
									}

								// Neumann boundary condition

								if (cell->face(face_no)->at_boundary() && (cell->face(face_no)->boundary_id() != 0))
								{
									for (unsigned int i = 0; i < fe_support_on_face[face_no].size(); ++i)
										for (unsigned int j = 0; j < fe_support_on_face[face_no].size(); ++j)
										{
											const unsigned int ii = fe_support_on_face[face_no][i];
											const unsigned int jj = fe_support_on_face[face_no][j];
											cell_matrix(ii, jj) +=
												( 0.5 * (beta_n + beta_n_abs) * tr_phi[i] * tr_phi[j]) * JxW;
										}

									if (cell->face(face_no)->boundary_id() == 1) {
										const double neumann_value = -exact_sol.gradient(quadrature_point) * normal_s;
										const double dirichlet_value = exact_sol.value(quadrature_point);
										for (unsigned int i = 0; i < fe_support_on_face[face_no].size(); ++i)
										{
											const unsigned int ii = fe_support_on_face[face_no][i];
											cell_vector(ii) += -tr_phi[i] * (
													nu * neumann_value + 0.5 * (beta_n - beta_n_abs) * dirichlet_value
													) * JxW;
										}
									}

								}

							}

							for (unsigned int i = 0; i < fe_cell.dofs_per_cell; ++i)
								for (unsigned int j = 0; j < fe_cell.dofs_per_cell; ++j)
								{
									// if R-faces else Q-faces
									if ((std::fabs(normal[0]-(1)) < 1e-12) || (std::fabs(normal[0]+(1)) < 1e-12)) {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
										ll_matrix(i, j) += (beta_s) * u_phi[i] * u_phi[j] * JxW;
#else
										ll_matrix(i, j) += 0.5 * (beta_n + beta_n_abs) * u_phi[i] * u_phi[j] * JxW;
#endif
									}
									else{
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
										ll_matrix(i, j) += (
												ip_stab * u_phi[i] * u_phi[j]
												- nu * u_phi_grad[i] * normal_s * u_phi[j]
												- nu * u_phi[i] * u_phi_grad[j] * normal_s
												+ (beta_s) * u_phi[i] * u_phi[j]
												) * JxW;
#else
										ll_matrix(i, j) += (
												ip_stab * u_phi[i] * u_phi[j]
												- nu * u_phi_grad[i] * normal_s * u_phi[j]
												- nu * u_phi[i] * u_phi_grad[j] * normal_s
												+ 0.5 * (beta_n + beta_n_abs) * u_phi[i] * u_phi[j]
												) * JxW;
#endif
									}
								}

							//AU = F-B{U_hat}, here is the -B{U_hat} part.
							//{U_hat} => trace_values
							if (trace_reconstruct)
								for (unsigned int i = 0; i < fe_cell.dofs_per_cell; ++i)
								{
									// tr_phi in lf_matrix replaced by trace_values
									//
									// if R-faces else Q-faces
									if ((std::fabs(normal[0]-(1)) < 1e-12) || (std::fabs(normal[0]+(1)) < 1e-12)) {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
										l_rhs(i) -=  (beta_n - beta_s) * u_phi[i] * trace_values[q] * JxW;
#else
										l_rhs(i) -=  0.5*(beta_n - beta_n_abs) * u_phi[i] * trace_values[q] * JxW;
#endif
									}
									else{
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
										l_rhs(i) -= (
												nu * u_phi_grad[i] * normal_s
												+ ((beta_n - beta_s) - ip_stab) * u_phi[i]
												) * trace_values[q] * JxW;
#else
										l_rhs(i) -= (
												nu * u_phi_grad[i] * normal_s
												+ (0.5*(beta_n - beta_n_abs) - ip_stab) * u_phi[i]
												) * trace_values[q] * JxW;
#endif
									}
								}
						}
					}

					ll_matrix.gauss_jordan();


					if (trace_reconstruct == false)
					{
						// tmp_mat = fl_mat * ll_mat^{-1}
						//         = -CA^{-1}
						fl_matrix.mmult(tmp_matrix, ll_matrix);
						// cel_vec = tmp_mat * l_rhs
						//         = -CA^{-1}F
						tmp_matrix.vmult_add(cell_vector, l_rhs);
						// cel_mat += tmp_mat * lf_mat
						//				 = (D)-CA^{-1}B
						tmp_matrix.mmult(cell_matrix, lf_matrix, true);
						cell->get_dof_indices(local_dof_indices);
					}
					else
					{
						// tmp_rhs = ll_mat * l_rhs
						//				 = A^{-1}(F-BV)
						ll_matrix.vmult(tmp_rhs, l_rhs);
						loc_cell->set_dof_values(tmp_rhs, local_owned_sol_cell);
					}

					// copy local to global
					if (trace_reconstruct == false)
					{
						constraints.distribute_local_to_global(cell_matrix,
								cell_vector,
								local_dof_indices,
								system_matrix,
								system_rhs);
					}

				}

			if (trace_reconstruct == false)
			{
				system_matrix.compress(VectorOperation::add);
				system_rhs.compress(VectorOperation::add);
			}
    }

	template <int dim>
		void SpaceTimeHDG<dim>::solve()
    {
			TimerOutput::Scope t(computing_timer, "solve");

			LA::MPI::Vector completely_distributed_sol(local_owned_dofs_facet,
					mpi_communicator);

			SolverControl cn;
			PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
			solver.set_symmetric_mode(false);
			solver.solve(system_matrix, completely_distributed_sol, system_rhs);


			constraints.distribute(completely_distributed_sol);

			local_relevant_sol_facet = completely_distributed_sol;

			assemble_system(true);

			// %%%%%%%% THESE ARE FOR ERROR ESTIMATION %%%%%%%%%%%%%%
			// we exchange data between processors updating those ghost
			// elements in the <code>local_relevant_sol_cell</code>
			// variable that have been written by other processors
			local_owned_sol_cell.compress(VectorOperation::insert);
			local_relevant_sol_cell = local_owned_sol_cell;

			if (refinement_mode == adaptive_refinement) {
				estimated_error_per_cell.reinit(triangulation.n_active_cells());
				const AdvectionVelocity<dim> advection_velocity;
				std::map<types::boundary_id, const Function<dim> *> boundary_functions;
				Solution<dim> solution_function(nu);
				RightHandSide<dim> right_hand_side(nu);
				boundary_functions[0] = &solution_function;
				boundary_functions[1] = &solution_function;

				SpacetimeHDGErrorEstimator<dim>::estimate(
						get_default_linear_mapping(dof_handler_cell.get_triangulation()),
						dof_handler_cell,
						dof_handler_facet,
						QGauss<dim-1>(fe_facet.degree+1),
						QGauss<dim>(fe_cell.degree+1),
						boundary_functions, // neumann bc
						&right_hand_side, // source term
						local_relevant_sol_cell,
						local_relevant_sol_facet,
						estimated_error_per_cell,
						nu,
						&advection_velocity,
						/*n_threads=*/numbers::invalid_unsigned_int,
						/*subdomain_id=*/numbers::invalid_subdomain_id);

				estimated_error =
					VectorTools::compute_global_error(triangulation, estimated_error_per_cell, VectorTools::L2_norm);
			}

		}

	template <int dim>
		void SpaceTimeHDG<dim>::calculate_errors()
		{
			Vector<float> difference_per_cell_l2(triangulation.n_active_cells());
      Vector<float> difference_per_cell_sh1(triangulation.n_active_cells());
      Vector<float> difference_per_cell_th1(triangulation.n_active_cells());
			Vector<float> difference_per_cell_dif_jump(triangulation.n_active_cells());
      Vector<float> difference_per_cell_adv_jump(triangulation.n_active_cells());
      Vector<float> difference_per_cell_neumann(triangulation.n_active_cells());

			const QGauss<dim> quadrature_formula(fe_facet.degree+2);
			const QGauss<dim-1> face_quadrature_formula(fe_facet.degree+2);

			const UpdateFlags local_flags(update_values |
					update_gradients |
					update_JxW_values |
					update_quadrature_points);

			const UpdateFlags local_face_flags(update_values |
					update_gradients |
					update_JxW_values);

			const UpdateFlags flags(update_values |
					update_normal_vectors |
					update_quadrature_points |
					update_JxW_values);


			std::vector<types::global_dof_index> dof_indices(fe_facet.dofs_per_cell);

			FEValues<dim> fe_values_local(fe_cell,quadrature_formula,local_flags);
			FEFaceValues<dim> fe_face_values_local(fe_cell,face_quadrature_formula,local_face_flags);
			FEFaceValues<dim> fe_face_values(fe_facet,face_quadrature_formula,flags);

			const AdvectionVelocity<dim> advection_velocity;
			const RightHandSide<dim> right_hand_side(nu);
			const Solution<dim> exact_sol(nu);

#if defined(SemiUpwinding) || defined(SemiCentreFlux)
			const QGauss<dim-1> face_quadrature_formula_betamax(2);
			const UpdateFlags flags_betamax(update_normal_vectors | update_quadrature_points);
			FEFaceValues<dim> fe_face_values_betamax(fe_facet,face_quadrature_formula_betamax,flags_betamax);
#endif
			const unsigned int n_q_points = fe_values_local.get_quadrature().size();
			const unsigned int n_face_q_points = fe_face_values_local.get_quadrature().size();

      double temp_err_l2 = 0;
      double temp_err_sh1 = 0;
      double temp_err_th1 = 0;
      double temp_err_dif_jump = 0;
      double temp_err_adv_jump = 0;
      double temp_err_neumann = 0;

			typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_facet.begin_active(), endc = dof_handler_facet.end();
			for(; cell!=endc; ++cell)
				if (cell->is_locally_owned())
				{
					typename DoFHandler<dim>::active_cell_iterator loc_cell (&triangulation, cell->level(), cell->index(), &dof_handler_cell);
					fe_values_local.reinit(loc_cell);
					const double hK = std::pow(cell->measure(),1.0/dim);

					std::vector<double> elem_sol_vals(quadrature_formula.size());
					fe_values_local.get_function_values(local_owned_sol_cell, elem_sol_vals);
					std::vector<Tensor<1,dim>> elem_sol_grads(quadrature_formula.size());
					fe_values_local.get_function_gradients(local_owned_sol_cell, elem_sol_grads);

					for (unsigned int q = 0; q < n_q_points; ++q)
					{
						/* const Tensor<1,dim> advection = advection_velocity.value(fe_values_local.quadrature_point(q)); */
						const double exact_value = exact_sol.value(fe_values_local.quadrature_point(q));
						const Tensor<1,dim> exact_grad_value = exact_sol.gradient(fe_values_local.quadrature_point(q));
						const double JxW = fe_values_local.JxW(q);
						temp_err_l2 += ((exact_value-elem_sol_vals[q])*(exact_value-elem_sol_vals[q])) * JxW;
						for (unsigned int c = 1; c < dim; ++c)
							temp_err_sh1 += (
									(exact_grad_value[c]-elem_sol_grads[q][c]) * (exact_grad_value[c]-elem_sol_grads[q][c])
									) * JxW;
						if (hK < nu)
							temp_err_th1 += hK * (
									(exact_grad_value[0]-elem_sol_grads[q][0]) * (exact_grad_value[0]-elem_sol_grads[q][0])
									) * JxW;
						else
							temp_err_th1 += hK * nu * (
									(exact_grad_value[0]-elem_sol_grads[q][0]) * (exact_grad_value[0]-elem_sol_grads[q][0])
									) * JxW;
					}
					difference_per_cell_l2(cell->active_cell_index())  = std::sqrt(temp_err_l2);
					difference_per_cell_sh1(cell->active_cell_index()) = std::sqrt(nu)*std::sqrt(temp_err_sh1);
					difference_per_cell_th1(cell->active_cell_index()) = std::sqrt(temp_err_th1);
					temp_err_l2  = 0;
					temp_err_sh1 = 0;
					temp_err_th1 = 0;

					for (const auto face_no : cell->face_indices()) {
						fe_face_values_local.reinit(loc_cell, face_no);
						fe_face_values.reinit(cell, face_no);
						std::vector<double> face_sol_vals_local(face_quadrature_formula.size());
						std::vector<double> face_sol_vals(face_quadrature_formula.size());
						fe_face_values_local.get_function_values(local_owned_sol_cell, face_sol_vals_local);
						fe_face_values.get_function_values(local_relevant_sol_facet, face_sol_vals);

						for (unsigned int q = 0; q < n_face_q_points; ++q) {
							const double JxW = fe_face_values.JxW(q);
							const Point<dim> quadrature_point = fe_face_values.quadrature_point(q);
							const Tensor<1,dim> advection = advection_velocity.value(quadrature_point);
							const Tensor<1,dim> normal = fe_face_values.normal_vector(q);
							// if R-faces else Q-faces
							if ((std::fabs(normal[0]-(1)) < 1e-12) || (std::fabs(normal[0]+(1)) < 1e-12))
								temp_err_adv_jump +=
									(face_sol_vals[q] - face_sol_vals_local[q])*(face_sol_vals[q] - face_sol_vals_local[q]) * JxW;
							else {
								temp_err_adv_jump += std::fabs(advection * normal) *
									(face_sol_vals[q] - face_sol_vals_local[q])*(face_sol_vals[q] - face_sol_vals_local[q]) * JxW;
								temp_err_dif_jump += nu/hK *
									(face_sol_vals[q] - face_sol_vals_local[q])*(face_sol_vals[q] - face_sol_vals_local[q]) * JxW;
							}

							// Neumann boundary err term
							if (cell->face(face_no)->at_boundary() && (cell->face(face_no)->boundary_id() != 0)) {
								const double exact_value = exact_sol.value(quadrature_point);
								temp_err_neumann += std::fabs(advection * normal) *
									(face_sol_vals[q] - exact_value)*(face_sol_vals[q] - exact_value) * JxW;
							}
						}
					}
					difference_per_cell_dif_jump(cell->active_cell_index())  = std::sqrt(temp_err_dif_jump);
					difference_per_cell_adv_jump(cell->active_cell_index())  = std::sqrt(temp_err_adv_jump);
					difference_per_cell_neumann(cell->active_cell_index())  = std::sqrt(temp_err_neumann);
					temp_err_dif_jump = 0;
					temp_err_adv_jump = 0;
					temp_err_neumann = 0;
				}

			const double l2_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_l2, VectorTools::L2_norm);
			const double sH1_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_sh1, VectorTools::L2_norm);
			const double tH1_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_th1, VectorTools::L2_norm);
			const double dif_jump_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_dif_jump, VectorTools::L2_norm);
			const double adv_jump_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_adv_jump, VectorTools::L2_norm);
			const double neumann_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_neumann, VectorTools::L2_norm);
      error_list[0] = (l2_error*l2_error);
      error_list[1] = (sH1_error*sH1_error);
      error_list[2] = (tH1_error*tH1_error);
      error_list[3] = (dif_jump_error*dif_jump_error);
      error_list[4] = (adv_jump_error*adv_jump_error);
      error_list[5] = (neumann_error*neumann_error);
		}


	template <int dim>
		void SpaceTimeHDG<dim>::output_results(const unsigned int cycle, const double nu)
		{
			int n = std::round(std::abs(std::log10(nu)));
			std::string filename;
#ifdef ROTATINGPULSE
			filename = "rot_pulse";
#elif defined(BOUNDARYLAYER)
			filename = "bnd_layer";
#elif defined(INTERIORLAYER)
			filename = "int_layer";
#endif
			filename += "_q" + Utilities::int_to_string(fe_facet.degree, 1);
			filename += "_n" + Utilities::int_to_string(n, 1);
			filename += "_l";

			DataOut<dim> data_out;
			std::vector<std::string> name(1, "solution");
			std::vector<DataComponentInterpretation::DataComponentInterpretation>
				comp_type(1, DataComponentInterpretation::component_is_scalar);
			data_out.add_data_vector(dof_handler_cell,
					local_owned_sol_cell,
					name,
					comp_type);

			Vector<float> subdomain(triangulation.n_active_cells());
			for (unsigned int i = 0; i < subdomain.size(); ++i)
				subdomain(i) = triangulation.locally_owned_subdomain();
			data_out.add_data_vector(subdomain, "subdomain");

			data_out.build_patches(fe_facet.degree);
			switch (refinement_mode) {
				case uniform_refinement:
					{
						data_out.write_vtu_with_pvtu_record(
								"./vtus_uniform/", filename, cycle, mpi_communicator, 1, 8);
						break;
					}

				case adaptive_refinement:
					{
						data_out.write_vtu_with_pvtu_record(
								"./vtus_adaptive/", filename, cycle, mpi_communicator, 1, 8);
						break;
					}
			}
		}


	template <int dim>
		void SpaceTimeHDG<dim>::run()
		{
			const unsigned int n_ranks =
				Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

			std::string DAT_header = "START DATE: " + Utilities::System::get_date() +
				", TIME: " + Utilities::System::get_time();
#ifdef ROTATINGPULSE
			std::string PRB_header = "Rotating Guassian Pulse Problem, ";
#elif defined(BOUNDARYLAYER)
			std::string PRB_header = "Boundary Layer Problem, ";
#elif defined(INTERIORLAYER)
			std::string PRB_header = "Interior Layer Problem, ";
#endif
			std::string MPI_header = "Running with " + std::to_string(n_ranks) +
				" MPI process" + (n_ranks > 1 ? "es" : "") + ", PETSc ";

			MPI_header = MPI_header + "sparse direct MUMPS solver";

			std::string SOL_header = "Finite element space: " + fe_facet.get_name() + ", " + fe_cell.get_name();
#if defined(SemiUpwinding)
			std::string MET_header = "Space-time IP-HDG, with semi-upwinding penalty";
#elif defined(SemiCentreFlux)
			std::string MET_header = "Space-time IP-HDG, with semi-centered-flux penalty";
#else
			std::string MET_header = "Space-time IP-HDG, with classic upwinding penalty";
#endif

      pcout << std::string(80, '=') << std::endl;
      pcout << DAT_header << std::endl;
      pcout << std::string(80, '-') << std::endl;

      pcout << PRB_header << "nu = " << nu << std::endl;
      pcout << MPI_header << std::endl;
      pcout << SOL_header << std::endl;
      pcout << MET_header << std::endl;
      pcout << std::string(80, '=') << std::endl;

			std::vector<long> rusage_history(num_cycle);
			std::vector<double> wall_time(num_cycle);
			std::vector<double> cpu_time(num_cycle);
			const double final_time = 1;
			double time_step_size = 0.1;
			for (unsigned int cycle = 0; cycle < num_cycle; ++cycle)
			{
				pcout << std::string(80, '-') << std::endl;
				pcout << "Cycle " << cycle + 1 << std::endl;
				pcout << std::string(80, '-') << std::endl;

				if (cycle == 0)
					init_coarse_mesh(time_step_size,final_time);
				else {
					refine_mesh(final_time);
				}

				Timer timer;
				pcout << "Set up system..." << std::endl;
				setup_system();
				pcout << "  Global mesh: \t"
					<< triangulation.n_global_active_cells() << " cells" << std::endl;
				pcout << "  DoFHandler: \t"
					<< dof_handler_facet.n_dofs() << " DoFs" << std::endl;
				assemble_system(false);
				rusage_history[cycle] = get_mem_usage()/1024.0;
				pcout << "  Mem usage: \t" << rusage_history[cycle] << " MB" << std::endl;
				solve();
				calculate_errors();
				if (ifoutput)
					output_results(cycle,nu);
				timer.stop();
				pcout << "  Done! (" << timer.wall_time() << "s)" << std::endl;
				wall_time[cycle] = timer.wall_time();
				cpu_time[cycle] = timer.cpu_time();
				timer.reset();
				for (unsigned int i = 0; i < error_list.size()-2; ++i) {
					if (i < 3 || i ==5)
						error_list[error_list.size()-2] += error_list[i];
					error_list[error_list.size()-1] += error_list[i];
				}
				for (unsigned int i = 0; i < error_list.size(); ++i)
					error_list[i] = std::sqrt(error_list[i]);
				pcout << "Output results..." << std::endl;
				pcout << "  Triple norm error: \t" << error_list[7] << std::endl;
				pcout << "  Estimated error: \t" << estimated_error << std::endl;
				convergence_table.add_value("cells", triangulation.n_global_active_cells());
				convergence_table.add_value("dofs", dof_handler_facet.n_dofs());
				convergence_table.add_value("L2", error_list[0]);
				convergence_table.add_value("sH1", error_list[1]);
				convergence_table.add_value("tH1", error_list[2]);
				convergence_table.add_value("dfjp", error_list[3]);
				convergence_table.add_value("adjp", error_list[4]);
				convergence_table.add_value("neum", error_list[5]);
				convergence_table.add_value("nojp", error_list[6]);
				convergence_table.add_value("tnorm", error_list[7]);
				convergence_table.add_value("est", estimated_error);
				convergence_table.add_value("eff", estimated_error/error_list[6]);
				convergence_table.add_value("efft", estimated_error/error_list[7]);
				for (unsigned int i = 0; i < error_list.size(); ++i)
					error_list[i] = 0.0;
			}

			pcout << std::string(130, '=') << std::endl;
			pcout << "Convergence History: " << std::endl;

			convergence_table.set_scientific("L2", true);
			convergence_table.set_precision("L2", 1);
			convergence_table.set_scientific("sH1", true);
			convergence_table.set_precision("sH1", 1);
			convergence_table.set_scientific("tH1", true);
			convergence_table.set_precision("tH1", 1);
			convergence_table.set_scientific("dfjp", true);
			convergence_table.set_precision("dfjp", 1);
			convergence_table.set_scientific("adjp", true);
			convergence_table.set_precision("adjp", 1);
			convergence_table.set_scientific("neum", true);
			convergence_table.set_precision("neum", 1);
			convergence_table.set_scientific("nojp", true);
			convergence_table.set_precision("nojp", 1);
			convergence_table.set_scientific("tnorm", true);
			convergence_table.set_precision("tnorm", 1);
			convergence_table.set_scientific("est", true);
			convergence_table.set_precision("est", 1);
			convergence_table.set_scientific("eff", true);
			convergence_table.set_precision("eff", 1);
			convergence_table.set_scientific("efft", false);
			convergence_table.set_precision("efft", 1);

			convergence_table.evaluate_convergence_rates(
					"L2", "dofs", ConvergenceTable::reduction_rate_log2, dim);
			convergence_table.evaluate_convergence_rates(
					"sH1", "dofs", ConvergenceTable::reduction_rate_log2, dim);
			convergence_table.evaluate_convergence_rates(
					"tH1", "dofs", ConvergenceTable::reduction_rate_log2, dim);
			convergence_table.evaluate_convergence_rates(
					"dfjp", "dofs", ConvergenceTable::reduction_rate_log2, dim);
			convergence_table.evaluate_convergence_rates(
					"adjp", "dofs", ConvergenceTable::reduction_rate_log2, dim);
			convergence_table.evaluate_convergence_rates(
					"neum", "dofs", ConvergenceTable::reduction_rate_log2, dim);
			convergence_table.evaluate_convergence_rates(
					"nojp", "dofs", ConvergenceTable::reduction_rate_log2, dim);
			convergence_table.evaluate_convergence_rates(
					"tnorm", "dofs", ConvergenceTable::reduction_rate_log2, dim);

			pcout << std::string(130, '-') << std::endl;
			if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
				convergence_table.write_text(std::cout);
			pcout << std::string(130, '=') << std::endl;

			computing_timer.print_summary();
			computing_timer.reset();
		}

} // end of namespace SpaceTimeAdvecDiffuIPH



int main(int argc, char** argv)
{
	using namespace dealii;

	const unsigned int dim = 3;

	int opt;
	int opt_n = 0;
	int num_cycle = 0;
	int degree = 1;
	SpaceTimeAdvecDiffuIPH::SpaceTimeHDG<dim>::RefinementMode refinement_mode = SpaceTimeAdvecDiffuIPH::SpaceTimeHDG<dim>::uniform_refinement;
	bool ifoutput = false;

	// Five command line options accepted:
	// n (nu), c (cycle), p (polynomial order), 
	// a (toggle on amr mode) and o (toggle on vtu generation)
	while ( (opt = getopt(argc, argv, "n:c:p:ao")) != -1 ) {
		switch ( opt ) {
			case 'n':
				opt_n = atoi(optarg);
				break;
			case 'c':
				num_cycle = atoi(optarg);
				break;
			case 'p':
				degree = atoi(optarg);
				break;
			case 'a':
				refinement_mode = SpaceTimeAdvecDiffuIPH::SpaceTimeHDG<dim>::adaptive_refinement;
				break;
			case 'o':
				ifoutput = true;
				break;
			case '?':  // unknown option...
				std::cerr << "Unknown option: '" << char(optopt) << "'!" << std::endl;
				break;
		}
	}

	// Hacky solution to suppress PETSc unused option warnings:
	argc = 1;
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	const double nu = std::pow(10, -opt_n);

	try
	{
		SpaceTimeAdvecDiffuIPH::SpaceTimeHDG<dim> hdg_problem (degree, nu, refinement_mode, num_cycle, ifoutput);
		hdg_problem.run ();
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl
			<< std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		std::cerr << "Exception on processing: " << std::endl
			<< exc.what() << std::endl
			<< "Aborting!" << std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl
			<< std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		std::cerr << "Unknown exception!" << std::endl
			<< "Aborting!" << std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		return 1;
	}

	return 0;
}

