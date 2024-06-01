/*
 * -------------------------------------------------------------------------------------
 *       Filename:  sthdg_error_estimator.templates.h
 *
 *    Description:  a posteriori error estimators for space-time
 *									(IP)HDG method for advection-diffusion problems
 *									on deforming domains
 *
 *        Created:  2023-11-01 09:45:50 AM
 *         Author:  Yuan (Greg) Wang
 *   Organization:  Univeristy of Waterloo, Applied Mathematics
 * -------------------------------------------------------------------------------------
 */

#ifndef sthdg_error_estimator_templates_h
#define sthdg_error_estimator_templates_h

#include <deal.II/base/config.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_face.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/distributed/tria_base.h>

#include "sthdg_error_estimator.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <regex>
#include <streambuf>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace internal {
	template<int dim, int spacedim, typename number>
		struct ParallelData
		{
			/*
			 * The finite elements to be used.
			 */
			const dealii::hp::FECollection<dim, spacedim> fe_local;

			/*
			 * Quadrature fomulas to be used for cells/faces
			 */
			const dealii::hp::QCollection<dim-1> face_quadrature;
			const dealii::hp::QCollection<dim> cell_quadrature;

			/*
			 * FEFaceValues objects to integrate over the faces of
			 * the current and potentially of neighbor cells.
			 *
			 * Step 51 (HDG in deal.II):
			 * "...Hanging nodes are handled in the same way as for
			 * continuous finite elements: For the face elements which
			 * only define degrees of freedom on the face, this process
			 * sets the solution on the refined side to coincide with the
			 * representation on the coarse side..."
			 */
			dealii::hp::FEValues<dim, spacedim> fe_values_cell;
			dealii::hp::FEFaceValues<dim, spacedim> fe_face_values_cell;
			dealii::hp::FEFaceValues<dim, spacedim> fe_face_values_facet;
			dealii::hp::FEFaceValues<dim, spacedim> fe_face_values_nbcell;
			dealii::hp::FESubfaceValues<dim, spacedim> fe_subface_values_cell;
			dealii::hp::FESubfaceValues<dim, spacedim> fe_subface_values_facet;

			/*
			 * Vectors to store the dg, hdg and gradient jump at the
			 * quadrature points.
			 * These vectors are not allocated inside the functions
			 * that use it, but rather globally, since memory
			 * allocation is slow, in particular in presence of
			 * multiple threads where synchronization makes things even
			 * slower. The index denotes the index of the quadrature
			 * point.
			 */
			std::vector<number> hdg_jump;
			std::vector<number> grad_jump;

			/** Vectors for the values and gradients of the finite
			 * element function on one cell.
			 *
			 * Let grad_values be a short name for <tt>grad u_h</tt>, where the
			 * index is the number of the quadrature point.
			 */
			std::vector<number> values;
			std::vector<Tensor<1, spacedim, number>> grad_values;

			/**
			 * The same vectors for a neighbor cell
			 */
			std::vector<Tensor<1, spacedim, number>> neighbor_grad_values;

			/**
			 * Vector for the facet
			 */
			std::vector<number> facet_values;

			/**
			 * The normal vectors of the finite element function on one face.
			 * The index denotes the index of the quadrature points.
			 */
			std::vector<Tensor<1, spacedim>> normal_vectors;

			/**
			 * Normal vectors of the opposing face.
			 */
			std::vector<Tensor<1, spacedim>> neighbor_normal_vectors;

			/**
			 * Values of coefficients in the jumps, if they are given.
			 * Index is the number of the quadrature point.
			 */
			std::vector<Tensor<1, dim>> coefficient_values;

			/**
			 * Array for the products of Jacobian determinants and weights of
			 * quadraturs points.
			 */
			std::vector<double> JxW_values;

			/**
			 * The subdomain id we are to care for.
			 */
			const types::subdomain_id subdomain_id;

			/**
			 * Some more references to input data to the
			 * KellyErrorEstimator::estimate() function.
			 */
			const std::map<types::boundary_id, const Function<spacedim, number> *> *neumann_bc;
			const Function<spacedim, number> *righthandside;
			const TensorFunction<1,spacedim> *coefficients;

			/**
			 * Constructor.
			 */
			template <class FE>
			ParallelData(
					const FE &fe_cell,
					const FE &fe_facet,
					const dealii::hp::QCollection<dim-1> &face_quadrature,
					const dealii::hp::QCollection<dim> &cell_quadrature,
					const dealii::hp::MappingCollection<dim, spacedim> &mapping,
					const types::subdomain_id subdomain_id,
					const std::map<types::boundary_id, const Function<spacedim, number> *> *neumann_bc,
					const Function<spacedim, number> *righthandside,
					const TensorFunction<1,spacedim> *coefficients);

		};

	/**
	 * Implementation of the constructor of the ParallelData.
	 */
	template <int dim, int spacedim, typename number>
  template <class FE>
	ParallelData<dim, spacedim, number>::ParallelData(
			const FE &fe_cell,
			const FE &fe_facet,
			const dealii::hp::QCollection<dim-1> &face_quadrature,
			const dealii::hp::QCollection<dim> &cell_quadrature,
			const dealii::hp::MappingCollection<dim, spacedim> &mapping,
			const types::subdomain_id subdomain_id,
			const std::map<types::boundary_id, const Function<spacedim, number> *> *neumann_bc,
			const Function<spacedim, number> *righthandside,
			const TensorFunction<1,spacedim> *coefficients)
	: fe_local(fe_cell, fe_facet),
	face_quadrature(face_quadrature),
	cell_quadrature(cell_quadrature),
	fe_values_cell(mapping,
			fe_local,
			cell_quadrature,
			update_values |
			update_gradients |
			update_hessians |
			update_JxW_values |
			update_quadrature_points),
	fe_face_values_cell(mapping,
			fe_local,
			face_quadrature,
			update_values |
			update_gradients |
			update_JxW_values |
			update_quadrature_points |
			update_normal_vectors),
	fe_face_values_facet(mapping,
			fe_local,
			face_quadrature,
			update_values |
			update_JxW_values |
			update_quadrature_points |
			update_normal_vectors),
	fe_face_values_nbcell(mapping,
			fe_local,
			face_quadrature,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_normal_vectors),
	fe_subface_values_cell(mapping,
			fe_local,
			face_quadrature,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_normal_vectors),
	fe_subface_values_facet(mapping,
			fe_local,
			face_quadrature,
			update_values | update_normal_vectors),
	hdg_jump(face_quadrature.max_n_quadrature_points()),
	grad_jump(face_quadrature.max_n_quadrature_points()),
	values(face_quadrature.max_n_quadrature_points()),
	grad_values(face_quadrature.max_n_quadrature_points()),
	neighbor_grad_values(face_quadrature.max_n_quadrature_points()),
	facet_values(face_quadrature.max_n_quadrature_points()),
	normal_vectors(face_quadrature.max_n_quadrature_points()),
	neighbor_normal_vectors(face_quadrature.max_n_quadrature_points()),
	coefficient_values(face_quadrature.max_n_quadrature_points()),
	JxW_values(face_quadrature.max_n_quadrature_points()),
	subdomain_id(subdomain_id),
	neumann_bc(neumann_bc),
	righthandside(righthandside),
	coefficients(coefficients)
	{}

	/**
	 * Copy data from the local_face_integrals map of a single ParallelData
	 * object into a global such map. This is the copier stage of a WorkStream
	 * pipeline.
	 */
  template <int dim, int spacedim>
  void
  copy_local_to_global(
    const std::map<typename DoFHandler<dim, spacedim>::face_iterator, double> &local_face_integrals,
    std::map<typename DoFHandler<dim, spacedim>::face_iterator, double> &face_integrals)
  {
    // now copy locally computed elements into the global map;
		// the for loop loops over local face integrals, which is
		// map between local faces and integrals on them;
		// for this map iterator p, p->first is the local face,
		// while p->second is the local face integral.
    for (typename
				std::map<typename DoFHandler<dim, spacedim>::face_iterator, double>::const_iterator
					p = local_face_integrals.begin();
					p != local_face_integrals.end(); ++p)
      {
        // double check that the element does not already exists in the
        // global map
        Assert(face_integrals.find(p->first) == face_integrals.end(), ExcInternalError());
				Assert(numbers::is_finite(p->second), ExcInternalError());
				Assert(p->second >= 0, ExcInternalError());

        face_integrals[p->first] = p->second;
      }
  }

  /**
   * Actually do the computation based on the evaluated gradients in
   * ParallelData.
   */
  template <int dim, int spacedim, typename number>
  double
  integrate_over_face(
    ParallelData<dim, spacedim, number> &parallel_data,
    const typename DoFHandler<dim, spacedim>::face_iterator &face,
    dealii::hp::FEFaceValues<dim, spacedim> &fe_face_values_cell,
		const double nu)
  {
    const unsigned int n_q_points = parallel_data.values.size();

    // now grad_values contains the following:
    // - for an internal face, grad_values=[grad u]
    // - for a neumann boundary face, grad_values=grad u
    // evaluated at one of the quadrature points

    // next we have to multiply this with the normal vector. Since we have
    // taken the difference of gradients for internal faces, we may chose
    // the normal vector of one cell, taking that of the neighbor would only
    // change the sign. We take the outward normal.

    parallel_data.normal_vectors =
      fe_face_values_cell.get_present_fe_values().get_normal_vectors();


		for (unsigned int p = 0; p < n_q_points; ++p) {
			parallel_data.grad_jump[p] = (parallel_data.grad_values[p] * parallel_data.normal_vectors[p]);
			parallel_data.hdg_jump[p] = parallel_data.values[p] - parallel_data.facet_values[p];
		}

		// this is truly hK when dt is (much) smaller than hK.
		const double hK = face->diameter();

		// both grad_values and neighbor_grad_values
		// have their first (time) component set to zero already
    if (face->at_boundary() == false) {
			// compute the jump in the gradients
			for (unsigned int p = 0; p < n_q_points; ++p) {
				parallel_data.grad_jump[p] += (parallel_data.neighbor_grad_values[p] *
						parallel_data.neighbor_normal_vectors[p]);
				parallel_data.grad_jump[p] *= std::sqrt(nu)*std::sqrt(hK);
			}
		}

		// if Q-face: compute factors only;
		// we are not doing factoring to R-faces.
		// beta_s-0.5*beta_n is 1.5 or 0.5 on R-faces, which is not
		// worth implementing.
		if (!((std::fabs(parallel_data.normal_vectors[0][0]-(1)) < 1e-12) || (std::fabs(parallel_data.normal_vectors[0][0]+(1)) < 1e-12))) {
			/* (1) make beta the coefficients Function;
			 * (2) evaluate coefficients(beta) at quadrature points;
			 * (3) find beta * n at quadrature points using normal vector
			 * data we already have;
			 * (4) find maximum of beta * n, i.e., beta_s;
			 * (5) find beta_s - 0.5*(beta*n), the actual
			 * coefficient_values.
			 */
			std::vector<double>	beta_flux(n_q_points);
			parallel_data.coefficients->value_list(
					fe_face_values_cell.get_present_fe_values()
					.get_quadrature_points(),
					parallel_data.coefficient_values);
			for (unsigned int p = 0; p < n_q_points; ++p)
				beta_flux[p] = parallel_data.coefficient_values[p]*parallel_data.normal_vectors[p];
			double beta_s = std::fabs(beta_flux[0]);
			for (unsigned int i = 1; i < beta_flux.size(); ++i) {
				if (std::fabs(beta_flux[i]) > beta_s)
					beta_s = std::fabs(beta_flux[i]);
			}

			for (unsigned int p = 0; p < n_q_points; ++p)
				parallel_data.hdg_jump[p] *=
					(std::sqrt(beta_s-0.5*beta_flux[p])
					 + std::sqrt(nu)/std::sqrt(hK)
					 + std::sqrt(std::sqrt(hK))/std::sqrt(nu));
					 /* + std::sqrt(hK)/std::sqrt(nu)); */
		}


		// neumann boundary face, i.e., Omega(0)
		// compute difference between cell solution and boundary function
    if (face->at_boundary() == true) {
			if (face->boundary_id() == 1) {
				const types::boundary_id boundary_id = face->boundary_id();

				Assert(parallel_data.neumann_bc->find(boundary_id) !=
						parallel_data.neumann_bc->end(),
						ExcInternalError());
				// get the values of the boundary function at the quadrature points
				std::vector<number> g(n_q_points);
				// set values of g using the point-wise values of the
				// function neumann_bc.
				parallel_data.neumann_bc->find(boundary_id)
					->second->value_list(fe_face_values_cell.get_present_fe_values()
							.get_quadrature_points(), g);

				for (unsigned int p = 0; p < n_q_points; ++p)
					parallel_data.grad_jump[p] = parallel_data.values[p] - g[p];
			} else {
				for (unsigned int p = 0; p < n_q_points; ++p)
					parallel_data.grad_jump[p] = 0;
			}
		}

    // now grad_jump contains the following:
    // - for an internal face, grad_jump=[du/dn]
    // - for a neumann boundary face, grad_jump=du/dn-g
    // evaluated at one of the quadrature points

    parallel_data.JxW_values =
      fe_face_values_cell.get_present_fe_values().get_JxW_values();

    // take the square of the grad_jump[i] for integration, and sum up
		double face_integral = 0;
		for (unsigned int p = 0; p < n_q_points; ++p)
			face_integral += (
					numbers::NumberTraits<number>::abs_square(parallel_data.grad_jump[p])
					+ numbers::NumberTraits<number>::abs_square(parallel_data.hdg_jump[p])) *
				parallel_data.JxW_values[p];

    return face_integral;
  }

  /**
	 * Actually do the computation on a face which has no hanging
	 * nodes (it is regular), i.e. either face is at boundary, or
	 * the other side's refinement level is the same as that of
	 * this side, then handle the integration of these both cases
	 * together.
   */
  template <typename InputVector, int dim, int spacedim>
  void
  integrate_over_regular_face(
    const InputVector &solution_cell,
    const InputVector &solution_facet,
    ParallelData<dim, spacedim, typename InputVector::value_type> & parallel_data,
    std::map<typename DoFHandler<dim, spacedim>::face_iterator, double> &local_face_integrals,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &facet_cell,
    const unsigned int face_no,
    dealii::hp::FEFaceValues<dim, spacedim> &fe_face_values_cell,
    dealii::hp::FEFaceValues<dim, spacedim> &fe_face_values_nbcell,
		dealii::hp::FEFaceValues<dim, spacedim> &fe_face_values_facet,
		const double nu)
  {
    const typename DoFHandler<dim, spacedim>::face_iterator face = cell->face(face_no);

		// initialize data of the restriction of this cell to the
		// present face;
		// last argument is the fe_index;
		// what's passed into this is
		// parallel_data.fe_face_values_cell from within function
		// estimate_one_cell.
    fe_face_values_cell.reinit(cell,
				face_no,
				numbers::invalid_unsigned_int,
				numbers::invalid_unsigned_int,
				0);
		// ...and
		// parallel_data.fe_face_values_facet from within function
		// estimate_one_cell.
    fe_face_values_facet.reinit(facet_cell,
				face_no,
				numbers::invalid_unsigned_int,
				numbers::invalid_unsigned_int,
				1);

    // get gradients of the finite element
    // function on this cell
		fe_face_values_cell.get_present_fe_values().get_function_gradients(solution_cell, parallel_data.grad_values);
		for (unsigned int p = 0; p < parallel_data.grad_values.size(); ++p) {
			parallel_data.grad_values[p][0] = 0.0;
		}
		fe_face_values_cell.get_present_fe_values().get_function_values(solution_cell, parallel_data.values);
		fe_face_values_facet.get_present_fe_values().get_function_values(solution_facet, parallel_data.facet_values);

		// internal face; integrate jump of gradient across this face
		if (face->at_boundary() == false)
		{
			Assert(cell->neighbor(face_no).state() == IteratorState::valid,
					ExcInternalError());

			const typename DoFHandler<dim, spacedim>::active_cell_iterator
				neighbor = cell->neighbor(face_no);

			// find which number the current face has relative to the
			// neighboring cell;
			// in other words, return other_face_no such that
			// cell->neighbor(face_no)->neighbor(other_face_no)==cell
			const unsigned int neighbor_neighbor = cell->neighbor_of_neighbor(face_no);
			Assert(neighbor_neighbor < GeometryInfo<dim>::faces_per_cell,
					ExcInternalError());

			// get restriction of finite element function of @p{neighbor} to the
			// common face
			fe_face_values_nbcell.reinit(neighbor,
					neighbor_neighbor,
					numbers::invalid_unsigned_int,
					numbers::invalid_unsigned_int,
					0);

			// get gradients on neighbor cell
			fe_face_values_nbcell.get_present_fe_values()
				.get_function_gradients(solution_cell, parallel_data.neighbor_grad_values);
			for (unsigned int p = 0; p < parallel_data.neighbor_grad_values.size(); ++p) {
				// exclude time derivative jump
				parallel_data.neighbor_grad_values[p][0] = 0.0;
			}

			parallel_data.neighbor_normal_vectors =
				fe_face_values_nbcell.get_present_fe_values().get_normal_vectors();
		}

    // now go to the generic function that does all the other things
    local_face_integrals[face] =
      integrate_over_face(parallel_data, face, fe_face_values_cell, nu);
  }

  /**
   * The same applies as for the function above, except that integration is
   * over face @p face_no of @p cell, where the respective neighbor is
   * refined, so that the integration is a bit more complex.
   */
  template <typename InputVector, int dim, int spacedim>
  void
  integrate_over_irregular_face(
    const InputVector &solution_cell,
    const InputVector &solution_facet,
    ParallelData<dim, spacedim, typename InputVector::value_type> & parallel_data,
    std::map<typename DoFHandler<dim, spacedim>::face_iterator, double> &local_face_integrals,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &facet_cell,
    const unsigned int face_no,
    dealii::hp::FEFaceValues<dim, spacedim>    &fe_face_values_nbcell,
    dealii::hp::FESubfaceValues<dim, spacedim> &fe_subface_values_cell,
    dealii::hp::FESubfaceValues<dim, spacedim> &fe_subface_values_facet,
		const double nu)
  {
    const auto neighbor = cell->neighbor(face_no);
    (void)neighbor; // suppress compiler unused variable warnings
    const auto face = cell->face(face_no);

    Assert(neighbor.state() == IteratorState::valid, ExcInternalError());
    Assert(face->has_children(), ExcInternalError());

    // set up a vector of the gradients of the finite element function on
    // this cell at the quadrature points
    //
		// let grad_values be a short name for [a grad u_h], where the index
		// is the number of the quadrature point

    // store which number @p{cell} has in the list of neighbors of
    // @p{neighbor}
    const unsigned int neighbor_neighbor = cell->neighbor_of_neighbor(face_no);
    Assert(neighbor_neighbor < GeometryInfo<dim>::faces_per_cell,
           ExcInternalError());

    // loop over all subfaces
    for (unsigned int subface_no = 0; subface_no < face->n_children();
         ++subface_no)
      {
        // get an iterator pointing to the cell behind the present subface
        const typename DoFHandler<dim, spacedim>::active_cell_iterator
          neighbor_child = cell->neighbor_child_on_subface(face_no, subface_no);
        Assert(neighbor_child->is_active(), ExcInternalError());

        // restrict the finite element on the present cell to the subface
        fe_subface_values_cell.reinit(cell,
                                 face_no,
                                 subface_no,
																 numbers::invalid_unsigned_int,
																 numbers::invalid_unsigned_int,
																 0);
        fe_subface_values_facet.reinit(facet_cell,
                                 face_no,
                                 subface_no,
																 numbers::invalid_unsigned_int,
																 numbers::invalid_unsigned_int,
																 1);

        // restrict the finite element on the neighbor cell to the common
        // @p{subface}.
        fe_face_values_nbcell.reinit(neighbor_child,
                              neighbor_neighbor,
															numbers::invalid_unsigned_int,
															numbers::invalid_unsigned_int,
															0);

        // store the gradient of the solution in grad_values
				fe_subface_values_cell.get_present_fe_values().get_function_gradients(
						solution_cell, parallel_data.grad_values);

				// store the values of the solution and the facet solution
				// on the current subface
				fe_subface_values_cell.get_present_fe_values().get_function_values(
						solution_cell, parallel_data.values);
				fe_subface_values_facet.get_present_fe_values().get_function_values(
						solution_facet, parallel_data.facet_values);

				// store the gradient from the neighbor's side in @p{neighbor_grad_values}
				fe_face_values_nbcell.get_present_fe_values().get_function_gradients(
						solution_cell, parallel_data.neighbor_grad_values);

        // call generic evaluate function
        parallel_data.neighbor_normal_vectors =
          fe_subface_values_cell.get_present_fe_values().get_normal_vectors();

        local_face_integrals[neighbor_child->face(neighbor_neighbor)] =
          integrate_over_face(parallel_data, face, fe_face_values_nbcell, nu); // THE 3RD ARG???
      }

    // finally loop over all subfaces to collect the contributions of the
    // subfaces and store them with the mother face
    double sum = 0;
    for (unsigned int subface_no = 0; subface_no < face->n_children(); ++subface_no)
      {
        Assert(local_face_integrals.find(face->child(subface_no)) !=
                 local_face_integrals.end(),
               ExcInternalError());
        Assert(local_face_integrals[face->child(subface_no)] >= 0,
               ExcInternalError());

				sum += local_face_integrals[face->child(subface_no)];
      }

    local_face_integrals[face] = sum;
  }

  /**
   * Computate the error on the faces of a single cell.
   */
  template <typename InputVector, int dim, int spacedim>
  void
  estimate_one_cell(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &facet_cell,
    ParallelData<dim, spacedim, typename InputVector::value_type> &parallel_data,
    std::map<typename DoFHandler<dim, spacedim>::face_iterator, double> &local_face_integrals,
    std::map<typename DoFHandler<dim, spacedim>::cell_iterator, double> &cell_integrals,
    const InputVector &solution_cell,
    const InputVector &solution_facet,
		const double nu)
  {
    const types::subdomain_id subdomain_id = parallel_data.subdomain_id;

		/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
		/* TEMPORARY Here we compute cell_integrals given one cell. */
		/* LambdaK unimplemented yet. */
		if (cell->subdomain_id() == subdomain_id) {
			const double hK = cell->diameter();
			parallel_data.fe_values_cell.reinit(cell,
					numbers::invalid_unsigned_int,
					numbers::invalid_unsigned_int,
					0);
			const unsigned int n_q_points = parallel_data.cell_quadrature.max_n_quadrature_points();
			std::vector<double> cell_residual(n_q_points);
			std::vector<double> cell_JxW(n_q_points);
			std::vector<Tensor<2,dim>> hessians_at_q_points(n_q_points);
			std::vector<Tensor<1,dim>> gradient_at_q_points(n_q_points);
			std::vector<Tensor<1,dim>> beta_values(n_q_points);
			std::vector<double> rhs_values(n_q_points);
			parallel_data.coefficients->value_list(
					parallel_data.fe_values_cell.get_present_fe_values()
					.get_quadrature_points(),
					beta_values);
			parallel_data.righthandside->value_list(
					parallel_data.fe_values_cell.get_present_fe_values()
					.get_quadrature_points(),
					rhs_values);
			parallel_data.fe_values_cell.get_present_fe_values().get_function_gradients(solution_cell, gradient_at_q_points);
			parallel_data.fe_values_cell.get_present_fe_values().get_function_hessians(solution_cell, hessians_at_q_points);
			cell_JxW = parallel_data.fe_values_cell.get_present_fe_values().get_JxW_values();

			for (unsigned int p = 0; p < n_q_points; ++p) {
				for (unsigned int d = 1; d < dim; ++d) {
					cell_residual[p] +=
						nu*hessians_at_q_points[p][d][d];
				}
				cell_residual[p] -= beta_values[p] * gradient_at_q_points[p];
				cell_residual[p] += rhs_values[p];
			}
			cell_integrals[cell] = 0;
			for (unsigned int p = 0; p < n_q_points; ++p)
				cell_integrals[cell] +=
					(numbers::NumberTraits<double>::abs_square(cell_residual[p])) *
					cell_JxW[p];
			cell_integrals[cell] *= std::min(1.0,hK*hK/(nu));
		}
		/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

    // empty our own copy of the local face integrals
    local_face_integrals.clear();

    // loop over all faces of this cell
    for (const unsigned int face_no : cell->face_indices())
      {
        const typename DoFHandler<dim, spacedim>::face_iterator face = cell->face(face_no);

				// make sure we do work only once: this face may either
				// be regular or irregular
				//
				// I. no work for regular interior faces with current cell
				// having bigger index
				if ((face->has_children() == false) // (1). face is not
																						// further refined on
																						// the neighbor side
						&& !cell->at_boundary(face_no)  // (2). face is not on the
																						// boundary
						&& (!cell->neighbor_is_coarser(face_no) // (3.1). face is not
																										// further refined
																										// compared to the
																										// neighbor side
							&& (cell->neighbor(face_no)->index() < cell->index() // (3.2). index of my cell
																																	 // is bigger than the
																																	 // neighbor side
								|| (cell->neighbor(face_no)->index() == cell->index()  // (3.3). this final "or"
																																			 // looks to cater
																																			 // for anisotropic
																																			 // settings
									&& cell->neighbor(face_no)->level() < cell->level()))))
          continue;

				// II. no work for irregular interior faces with current
				// cell more refined (we integrate over the subfaces when
				// we visit the coarse cells)
        if (face->at_boundary() == false)
          if (cell->neighbor_is_coarser(face_no))
            continue;

				// III. no work for non-neumann boundary faces
				// (however, to make things easier when summing up the
				// contributions of the faces of cells, we enter this
				// face into the list of faces with contribution zero)
        if (face->at_boundary() &&
            (parallel_data.neumann_bc->find(face->boundary_id()) ==
             parallel_data.neumann_bc->end()))
          {
            local_face_integrals[face] = 0.;
            continue;
          }

				// IV. no work for faces when none of the cells relevant
				// to current face is in the subdomain we care
        if (!((subdomain_id == numbers::invalid_subdomain_id) ||
							(cell->subdomain_id() == subdomain_id)))
          {
            // ok, cell is unwanted, but maybe its neighbor behind the face
            // we presently work on? oh is there a face at all?
            if (face->at_boundary())
              continue;

            bool care_for_cell = false;
            if (face->has_children() == false)
              care_for_cell |=
                ((cell->neighbor(face_no)->subdomain_id() == subdomain_id) ||
                 (subdomain_id == numbers::invalid_subdomain_id));
            else
              {
                for (unsigned int sf = 0; sf < face->n_children(); ++sf)
									if ((cell->neighbor_child_on_subface(face_no, sf)
												->subdomain_id() == subdomain_id) ||
                      (subdomain_id == numbers::invalid_subdomain_id))
                    {
                      care_for_cell = true;
                      break;
                    }
              }

						// so if none of the neighbors cares for this
						// subdomain, then try next face
            if (care_for_cell == false)
              continue;
          }

				// so now we know that we care for this face:
				// (1). regular interior faces with current cell having smaller index;
				// (2). irregular interior faces with current cell being coarser;
				// (3). neumann boundary faces
        if (face->has_children() == false)
					// handles (1) and (3)
          integrate_over_regular_face(solution_cell,
																			solution_facet,
                                      parallel_data,
                                      local_face_integrals,
                                      cell,
																			facet_cell,
                                      face_no,
                                      parallel_data.fe_face_values_cell,
                                      parallel_data.fe_face_values_nbcell,
                                      parallel_data.fe_face_values_facet,
																			nu);

        else
					// handles (2)
          integrate_over_irregular_face(solution_cell,
																				solution_facet,
                                        parallel_data,
                                        local_face_integrals,
                                        cell,
                                        facet_cell,
                                        face_no,
                                        parallel_data.fe_face_values_nbcell,
                                        parallel_data.fe_subface_values_cell,
                                        parallel_data.fe_subface_values_facet,
																				nu);
      }
  }

}

template <int dim, int spacedim>
template <typename InputVector>
void
SpacetimeHDGErrorEstimator<dim, spacedim>::estimate(
  const Mapping<dim, spacedim> &   mapping,
  const DoFHandler<dim, spacedim> &dof_handler_cell,
  const DoFHandler<dim, spacedim> &dof_handler_facet,
	const Quadrature<dim-1>  &face_quadrature,
	const Quadrature<dim>  &cell_quadrature,
  const std::map<types::boundary_id,
                 const Function<spacedim, typename InputVector::value_type> *> &neumann_bc,
	const Function<spacedim, typename InputVector::value_type> *righthandside,
  const InputVector &solution_cell,
  const InputVector &solution_facet,
  Vector<double> &error,
	const double nu,
  const TensorFunction<1,dim> *coefficients,
  const unsigned int n_threads,
  const types::subdomain_id subdomain_id_)
{
	estimate(mapping,
			dof_handler_cell,
			dof_handler_facet,
			hp::QCollection<dim-1>(face_quadrature),
			hp::QCollection<dim>(cell_quadrature),
			neumann_bc,
			righthandside,
			solution_cell,
			solution_facet,
			error,
			nu,
			coefficients,
			n_threads,
			subdomain_id_);
}

template <int dim, int spacedim>
template <typename InputVector>
void
SpacetimeHDGErrorEstimator<dim, spacedim>::estimate(
  const Mapping<dim, spacedim> &   mapping,
  const DoFHandler<dim, spacedim> &dof_handler_cell,
  const DoFHandler<dim, spacedim> &dof_handler_facet,
	const hp::QCollection<dim-1>  &face_quadrature,
	const hp::QCollection<dim>  &cell_quadrature,
  const std::map<types::boundary_id,
                 const Function<spacedim, typename InputVector::value_type> *> &neumann_bc,
	const Function<spacedim, typename InputVector::value_type> *righthandside,
  const InputVector &solution_cell,
  const InputVector &solution_facet,
  Vector<double> &error,
	const double nu,
  const TensorFunction<1,dim> *coefficients,
  const unsigned int n_threads,
  const types::subdomain_id subdomain_id_)
{
	(void)n_threads; // suppress compiler unused variable warnings
  types::subdomain_id subdomain_id = numbers::invalid_subdomain_id;
  if (const auto *triangulation = dynamic_cast<
        const parallel::DistributedTriangulationBase<dim, spacedim> *>(
        &dof_handler_cell.get_triangulation()))
    {
			/* std::cout << triangulation->locally_owned_subdomain() << std::endl; */
      Assert((subdomain_id_ == numbers::invalid_subdomain_id) ||
               (subdomain_id_ == triangulation->locally_owned_subdomain()),
             ExcMessage(
               "For distributed Triangulation objects and associated "
               "DoFHandler objects, asking for any subdomain other than the "
               "locally owned one does not make sense."));
      subdomain_id = triangulation->locally_owned_subdomain();
    }
  else
    {
      subdomain_id = subdomain_id_;
    }

	// Map of integrals indexed by the corresponding face. In this
	// map we store the integrated jump of the gradient for each
	// face.  At the end of the function, we again loop over the
	// cells and collect the contributions of the different faces
	// of the cell.
  std::map<typename DoFHandler<dim, spacedim>::face_iterator, double> face_integrals;
	// copier not needed since cell integrals are independent of
	// each other.
	// NB: our test case has a zero right-hand side.
  std::map<typename DoFHandler<dim, spacedim>::cell_iterator, double> cell_integrals;

  const hp::MappingCollection<dim, spacedim> mapping_collection(mapping);
  const internal::ParallelData<dim, spacedim, typename InputVector::value_type>
    parallel_data(dof_handler_cell.get_fe(),
				dof_handler_facet.get_fe(),
				face_quadrature,
				cell_quadrature,
				mapping_collection,
				subdomain_id,
				&neumann_bc,
				righthandside,
				coefficients);
  std::map<typename DoFHandler<dim, spacedim>::face_iterator, double> sample_local_face_integrals;

  // now let's work on all those cells:
  WorkStream::run(
    dof_handler_cell.begin_active(),
    static_cast<typename DoFHandler<dim, spacedim>::active_cell_iterator>(dof_handler_cell.end()),
    [&solution_cell, &solution_facet, &dof_handler_facet, nu, &cell_integrals](
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      internal::ParallelData<dim, spacedim, typename InputVector::value_type> & parallel_data,
      std::map<typename DoFHandler<dim, spacedim>::face_iterator, double> &local_face_integrals) {
			typename DoFHandler<dim>::active_cell_iterator facet_cell(&dof_handler_facet.get_triangulation(), cell->level(), cell->index(), &dof_handler_facet);
      internal::estimate_one_cell(cell, facet_cell, parallel_data, local_face_integrals, cell_integrals, solution_cell, solution_facet, nu);
    },
    [&face_integrals](
      const std::map<typename DoFHandler<dim, spacedim>::face_iterator, double> &local_face_integrals) {
      internal::copy_local_to_global<dim, spacedim>(local_face_integrals, face_integrals);
    },
    parallel_data,
    sample_local_face_integrals);

  // finally add up the contributions of the faces for each cell

  // reserve one slot for each cell and set it to zero

  error.reinit(dof_handler_cell.get_triangulation().n_active_cells());
  for (unsigned int i = 0; i < dof_handler_cell.get_triangulation().n_active_cells(); ++i)
    error(i) = 0;

  // now walk over all cells and collect information from the faces. only do
  // something if this is a cell we care for based on the subdomain id
  for (const auto &cell : dof_handler_cell.active_cell_iterators())
		if ((subdomain_id == numbers::invalid_subdomain_id) ||
				(cell->subdomain_id() == subdomain_id))
			//	For programs that are parallelized using MPI but where
			//	meshes are held distributed across several processors
			//	using the parallel::distributed::Triangulation class, the
			//	subdomain id of cells is tied to the processor that owns
			//	the cell. In other words, querying the subdomain id of a
			//	cell tells you if the cell is owned by the current
			//	processor (i.e. if cell->subdomain_id() ==
			//	triangulation.parallel::distributed::Triangulation::locally_owned_subdomain())
			//	or by another processor. In the parallel distributed
			//	case, subdomain ids are only assigned to cells that the
			//	current processor owns as well as the immediately
			//	adjacent ghost cells. Cells further away are held on each
			//	processor to ensure that every MPI process has access to
			//	the full coarse grid as well as to ensure the invariant
			//	that neighboring cells differ by at most one refinement
			//	level. These cells are called "artificial" and have the
			//	special subdomain id value types::artificial_subdomain_id.
      {
        const unsigned int present_cell = cell->active_cell_index();

        // loop over all faces of this cell
        for (const unsigned int face_no : cell->face_indices())
          {
            Assert(face_integrals.find(cell->face(face_no)) !=
                     face_integrals.end(),
                   ExcInternalError());

						// make sure that we have written a meaningful value into this
						// slot
						Assert(face_integrals[cell->face(face_no)] >= 0, ExcInternalError());

						error(present_cell) += face_integrals[cell->face(face_no)];
          }
				error(present_cell) += cell_integrals[cell];
				error(present_cell) = std::sqrt(error(present_cell));
      }
}

DEAL_II_NAMESPACE_CLOSE

#endif
