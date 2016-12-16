/*
 *  gif2_psc_exp.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/*
  This version uses values of V_m for the solver that are relative to E_L.
*/


#include "gif2_psc_exp.h"

#ifdef HAVE_GSL

// C++ includes:
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "nest_names.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap< nest::gif2_psc_exp >
  nest::gif2_psc_exp::recordablesMap_;

namespace nest
{
/*
 * template specialization must be placed in namespace
 *
 * Override the create() method with one call to RecordablesMap::insert_()
 * for each quantity to be recorded.
 */
template <>
void
RecordablesMap< gif2_psc_exp >::create()
{
  // use standard names whereever you can for consistency!
  insert_(
    names::V_m, &gif2_psc_exp::get_y_elem_< gif2_psc_exp::State_::V_M > );
  insert_( names::w, &gif2_psc_exp::get_y_elem_< gif2_psc_exp::State_::W > );
  insert_(
    names::I_syn_ex, &gif2_psc_exp::get_y_elem_< gif2_psc_exp::State_::I_syn_ex > );
  insert_(
    names::I_syn_in, &gif2_psc_exp::get_y_elem_< gif2_psc_exp::State_::I_syn_in > );

}
}


extern "C" int
nest::gif2_psc_exp_dynamics( double,
  const double y[],
  double f[],
  void* pnode )
{
  // a shorthand
  typedef nest::gif2_psc_exp::State_ S;

  // get access to node so we can almost work as in a member function
  assert( pnode );
  const nest::gif2_psc_exp& node =
    *( reinterpret_cast< nest::gif2_psc_exp* >( pnode ) );

  // y[] here is---and must be---the state vector supplied by the integrator,
  // not the state vector in the node, node.S_.y[].
  // The following code is verbose for the sake of clarity. We assume that a
  // good compiler will optimize the verbosity away ...

  // shorthand for state variables
  const double_t& V = y[ S::V_M ];
  const double_t& w = y[ S::W ];
  const double_t I_syn_ex = y[ S::I_syn_ex];
  const double_t I_syn_in = y[ S::I_syn_in];

  // dv/dt
  f[ S::V_M ] =
    ( -node.P_.g * V
      - w * node.P_.g_rr
      + I_syn_ex + I_syn_in
      + node.P_.I_e + node.B_.I_stim_ ) / node.P_.C_m;

  // Adaptation current w.
  f[ S::W ] = ( V - w ) / node.P_.tau_1;

  // Synaptic currents
  f[ S::I_syn_ex ] = - I_syn_ex / node.P_.tau_syn_ex;
  f[ S::I_syn_in ] = - I_syn_in / node.P_.tau_syn_in;

  return GSL_SUCCESS;
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */
nest::gif2_psc_exp::Parameters_::Parameters_()
  : V_reset_( -56.0 - E_L ) // mV
  , t_ref_( 2.0 )     // ms
  , g ( 12.5 )        // membrane conductance, in nS, = C/tau_m
  , g_rr ( 25.0 )     // auxiliary variable conductance, in nS, = C/tau_v
  , C_m( 250.0 )      // pF
  , E_L( -70.0 )      // mV
  , tau_1( 10.0 )     // ms
  , V_th( -50.0 - E_L ) // mV
  , tau_syn_ex( 2.0 ) // ms
  , tau_syn_in( 2.0 ) // ms
  , I_e( 0.0 )        // pA
  , gsl_error_tol( 1e-6 ) // gsl throws if the tolerance is crossed
{
}

nest::gif2_psc_exp::State_::State_( const Parameters_& p )
  : r_( 0 )
{
  y_[ 0 ] = p.E_L;
  for ( size_t i = 1; i < STATE_VEC_SIZE; ++i )
    y_[ i ] = 0;
}

nest::gif2_psc_exp::State_::State_( const State_& s )
  : r_( s.r_ )
{
  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
    y_[ i ] = s.y_[ i ];
}

nest::gif2_psc_exp::State_& nest::gif2_psc_exp::State_::operator=(
  const State_& s )
{
  assert( this != &s ); // would be bad logical error in program

  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
    y_[ i ] = s.y_[ i ];
  r_ = s.r_;
  return *this;
}

/* ----------------------------------------------------------------
 * Paramater and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
nest::gif2_psc_exp::Parameters_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::C_m, C_m );
  def< double >( d, names::V_th, V_th + E_L );
  def< double >( d, names::t_ref, t_ref_ );
  def< double >( d, names::g, g );
  def< double >( d, names::g_rr, g_rr );
  def< double >( d, names::E_L, E_L );
  def< double >( d, names::V_reset, V_reset_ + E_L );
  def< double >( d, names::tau_syn_ex, tau_syn_ex );
  def< double >( d, names::tau_syn_in, tau_syn_in );
  def< double >( d, names::tau_1, tau_1 );
  def< double >( d, names::I_e, I_e );
  def< double >( d, names::gsl_error_tol, gsl_error_tol );
}

double
nest::gif2_psc_exp::Parameters_::set( const DictionaryDatum& d )
{
  // update everything that depends on E_L for possible changes
  const double ELold = E_L;
  updateValue< double >( d, names::E_L, E_L );
  const double delta_EL = E_L - ELold;
  if ( updateValue< double >( d, names::V_reset, V_reset_ ) )
    V_reset_ -= E_L;
  else
    V_reset_ -= delta_EL;

  if ( updateValue< double >( d, names::V_th, V_th ) )
    V_th -= E_L;
  else
    V_th -= delta_EL;
  // now update everything else
  updateValue< double >( d, names::t_ref, t_ref_ );
  updateValue< double >( d, names::C_m, C_m );
  updateValue< double >( d, names::g, g );
  updateValue< double >( d, names::g_rr, g_rr );
  updateValue< double >( d, names::tau_syn_ex, tau_syn_ex );
  updateValue< double >( d, names::tau_syn_in, tau_syn_in );
  updateValue< double >( d, names::tau_1, tau_1 );
  updateValue< double >( d, names::I_e, I_e );
  updateValue< double >( d, names::gsl_error_tol, gsl_error_tol );

  // let's check if everything is consistent
  if ( V_reset_ >= V_th )
    throw BadProperty( "V_reset must be lower than threshold." );

  if ( C_m <= 0 )
    throw BadProperty( "Ensure that C_m >0" );

  if ( t_ref_ < 0 )
    throw BadProperty( "Ensure that t_ref >= 0" );

  if ( tau_syn_ex <= 0 || tau_syn_in <= 0 || tau_1 <= 0 )
    throw BadProperty( "All time constants must be strictly positive." );

  if ( gsl_error_tol <= 0. )
    throw BadProperty( "The gsl_error_tol must be strictly positive." );

  //inform on the change in E_L
  return delta_EL;
}

void
nest::gif2_psc_exp::State_::get( DictionaryDatum& d, const Parameters_& p ) const
{
  def< double >( d, names::V_m, y_[ V_M ] + p.E_L );
  def< double >( d, names::w, y_[ W ] );
  def< double >( d, names::I_syn_ex, y_[ I_syn_ex ] );
  def< double >( d, names::I_syn_in, y_[ I_syn_in ] );
}

void
nest::gif2_psc_exp::State_::set(const DictionaryDatum& d
    , const Parameters_& p
    , double delta_EL )
{
  if ( updateValue< double >( d, names::V_m, y_[ V_M ] ) )
    y_[ V_M ] -= p.E_L;
  else
    y_[ V_M ] -= delta_EL;
  updateValue< double >( d, names::w, y_[ W ] );
  updateValue< double >( d, names::I_syn_ex, y_[ I_syn_ex ] );
  updateValue< double >( d, names::I_syn_in, y_[ I_syn_in ] );
}

nest::gif2_psc_exp::Buffers_::Buffers_( gif2_psc_exp& n )
  : logger_( n )
  , s_( 0 )
  , c_( 0 )
  , e_( 0 )
{
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

nest::gif2_psc_exp::Buffers_::Buffers_( const Buffers_&, gif2_psc_exp& n )
  : logger_( n )
  , s_( 0 )
  , c_( 0 )
  , e_( 0 )
{
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */

nest::gif2_psc_exp::gif2_psc_exp()
  : Archiving_Node()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create();
}

nest::gif2_psc_exp::gif2_psc_exp( const gif2_psc_exp& n )
  : Archiving_Node( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

nest::gif2_psc_exp::~gif2_psc_exp()
{
  // GSL structs may not have been allocated, so we need to protect destruction
  if ( B_.s_ )
    gsl_odeiv_step_free( B_.s_ );
  if ( B_.c_ )
    gsl_odeiv_control_free( B_.c_ );
  if ( B_.e_ )
    gsl_odeiv_evolve_free( B_.e_ );
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
nest::gif2_psc_exp::init_state_( const Node& proto )
{
  const gif2_psc_exp& pr = downcast< gif2_psc_exp >( proto );
  S_ = pr.S_;
}

void
nest::gif2_psc_exp::init_buffers_()
{
  B_.spike_exc_.clear(); // includes resize
  B_.spike_inh_.clear(); // includes resize
  B_.currents_.clear();  // includes resize
  Archiving_Node::clear_history();

  B_.logger_.reset();

  B_.step_ = Time::get_resolution().get_ms();

  // We must integrate this model with high-precision to obtain decent results
  B_.IntegrationStep_ = std::min( 0.01, B_.step_ );

  if ( B_.s_ == 0 )
  // pointer to a newly allocated instance of a stepping function s_
    B_.s_ =
      gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, State_::STATE_VEC_SIZE );
  else
  // resets stepping function s_ when it is not continued
    gsl_odeiv_step_reset( B_.s_ );

  if ( B_.c_ == 0 )
  // c_ controls local errors on each step abs/rel'ly wrt the solution y_i(t).
    B_.c_ = gsl_odeiv_control_yp_new( P_.gsl_error_tol, P_.gsl_error_tol );
  else
  // initialise c_ with errors and scaling factors
    gsl_odeiv_control_init(
      B_.c_, P_.gsl_error_tol, P_.gsl_error_tol, 0.0, 1.0 );

  if ( B_.e_ == 0 )
  // pointer to newly allocated evolution function of arg size
    B_.e_ = gsl_odeiv_evolve_alloc( State_::STATE_VEC_SIZE );
  else
  // reset evolution function, when it is not continued
    gsl_odeiv_evolve_reset( B_.e_ );

  B_.sys_.function = gif2_psc_exp_dynamics;
  B_.sys_.jacobian = NULL;
  B_.sys_.dimension = State_::STATE_VEC_SIZE;
  B_.sys_.params = reinterpret_cast< void* >( this );

  // This variable is the applied current. Must be buffer to be forwarded
  // to the GSL routines
  B_.I_stim_ = 0.0;
}

void
nest::gif2_psc_exp::calibrate()
{
  // ensures initialization in case mm connected after Simulate
  B_.logger_.init();
  V_.RefractoryCounts_ = Time( Time::ms( P_.t_ref_ ) ).get_steps();
  // since t_ref_ >= 0, this can only fail in error
  assert( V_.RefractoryCounts_ >= 0 );
}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void
nest::gif2_psc_exp::update( const Time& origin,
  const long from,
  const long to )
{
  assert(
    to >= 0 && ( delay ) from < kernel().connection_manager.get_min_delay() );
  assert( from < to );
  assert( State_::V_M == 0 );

  for ( long lag = from; lag < to; ++lag )
  {
    // Richardson et al. (2003) used Runge-Kutta second order with 10/20microsec.
    // We use the RK45 from gsl instead.

    /*
    1.) Evolve system for current timestep
    2.) Spike handling for next step
    3.) Read new input current
    */

    // time index for the solver-while loop
    double t = 0.0;

    if ( S_.r_ > 0 ) // refractory?
      --S_.r_;

    // numerical integration with adaptive step size control:
    // ------------------------------------------------------
    // gsl_odeiv_evolve_apply performs only a single numerical
    // integration step, starting from t and bounded by step;
    // the while-loop ensures integration over the whole simulation
    // step (0, step] if more than one integration step is needed due
    // to a small integration step size;
    // note that (t+IntegrationStep > step) leads to integration over
    // (t, step] and afterwards setting t to step, but it does not
    // enforce setting IntegrationStep to step-t
    while ( t < B_.step_ )
    {
      const int status = gsl_odeiv_evolve_apply(
        B_.e_,
        B_.c_,
        B_.s_,
        &B_.sys_,             // system of ODE
        &t,                   // from t
        B_.step_,             // to t <= step
        &B_.IntegrationStep_, // integration step size
        S_.y_ );              // neuronal state

      if ( status != GSL_SUCCESS )
        throw GSLSolverFailure( get_name(), status );
    }


    if ( S_.r_ > 0 ) // refractory -> keep reset current
      S_.y_[ State_::V_M ] = P_.V_reset_;
    else if ( S_.y_[ State_::V_M ] >= P_.V_th ) // Threshold crossed. Fire!
    {
      S_.y_[ State_::V_M ] = P_.V_reset_; // do not reset w
      S_.r_ = V_.RefractoryCounts_;
      set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );
      SpikeEvent se;
      kernel().event_delivery_manager.send( *this, se, lag );
    }

    // weighted instant spike effects:
    S_.y_[ State_::I_syn_ex ] += B_.spike_exc_.get_value( lag );
    S_.y_[ State_::I_syn_in ] += B_.spike_inh_.get_value( lag );

    // set new input current
    B_.I_stim_ = B_.currents_.get_value( lag );

    // log state data
    B_.logger_.record_data( origin.get_steps() + lag );
  }
}

void
nest::gif2_psc_exp::handle( SpikeEvent& e )
{
  assert( e.get_delay() > 0 );

  if ( e.get_weight() > 0.0 )
    B_.spike_exc_.add_value( e.get_rel_delivery_steps(
                               kernel().simulation_manager.get_slice_origin() ),
      e.get_weight() * e.get_multiplicity() );
  else
    B_.spike_inh_.add_value( e.get_rel_delivery_steps(
                               kernel().simulation_manager.get_slice_origin() ),
      e.get_weight() * e.get_multiplicity() ); // psc's instead of conductances
}

void
nest::gif2_psc_exp::handle( CurrentEvent& e )
{
  assert( e.get_delay() > 0 );

  const double_t c = e.get_current();
  const double_t w = e.get_weight();

  // current weights
  B_.currents_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    w * c );
}

void
nest::gif2_psc_exp::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

#endif // HAVE_GSL
