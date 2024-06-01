/*
 * -------------------------------------------------------------------------------------
 *       Filename:  sthdg_error_estimator.h
 *
 *    Description:  a posteriori error estimators for space-time
 *									(IP)HDG method for advection-diffusion problems 
 *									on deforming domains
 *
 *        Created:  2023-11-06 10:26:32 AM
 *         Author:  Yuan (Greg) Wang
 *   Organization:  Univeristy of Waterloo, Applied Mathematics
 * -------------------------------------------------------------------------------------
 */

#ifndef sthdg_error_estimator_h
#define sthdg_error_estimator_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <map>

DEAL_II_NAMESPACE_OPEN

// Forward declarations
template <int, int>
class DoFHandler;
template <int, int>
class Mapping;
template <int>
class Quadrature;

template <int dim, int spacedim = dim>
class SpacetimeHDGErrorEstimator
{
public:

  template <typename InputVector>
  static void
  estimate(
    const Mapping<dim, spacedim> &   mapping,
    const DoFHandler<dim, spacedim> &dof_cell,
    const DoFHandler<dim, spacedim> &dof_facet,
		const hp::QCollection<dim-1> &face_quadrature,
		const hp::QCollection<dim>	 &cell_quadrature,
		const std::map<types::boundary_id, const Function<spacedim, typename InputVector::value_type> *> & neumann_bc,
		const Function<spacedim, typename InputVector::value_type> *righthandside,
    const InputVector &       solution_cell,
    const InputVector &       solution_facet,
    Vector<double> &           error,
		const double nu,
    const TensorFunction<1,dim> *coefficients = nullptr, // also the advection field
    const unsigned int        n_threads      = numbers::invalid_unsigned_int,
    const types::subdomain_id subdomain_id   = numbers::invalid_subdomain_id);

  template <typename InputVector>
  static void
  estimate(
    const Mapping<dim, spacedim> &   mapping,
    const DoFHandler<dim, spacedim> &dof_cell,
    const DoFHandler<dim, spacedim> &dof_facet,
		const Quadrature<dim-1>  &face_quadrature,
		const Quadrature<dim>  &cell_quadrature,
    const std::map<types::boundary_id, const Function<spacedim, typename InputVector::value_type> *> & neumann_bc,
		const Function<spacedim, typename InputVector::value_type> *righthandside,
    const InputVector &       solution_cell,
    const InputVector &       solution_facet,
    Vector<double> &           error,
		const double nu,
    const TensorFunction<1,dim> *coefficients = nullptr,// also the advection field
    const unsigned int        n_threads      = numbers::invalid_unsigned_int,
    const types::subdomain_id subdomain_id   = numbers::invalid_subdomain_id);

};


DEAL_II_NAMESPACE_CLOSE

#endif
