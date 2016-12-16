/*
 *  gif2_psc_exp.h
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

#ifndef GIF2_PSC_EXP_H
#define GIF2_PSC_EXP_H

// Generated includes:
#include "config.h"

#ifdef HAVE_GSL

// External includes:
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "recordables_map.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

/* BeginDocumentation
Name: gif2_psc_exp - 2-variable generalised integrate-and-fire

Description:

gif2_psc_exp is an implementation of a generalised leaky integrate-and-fire model
with exponentially decaying synaptic currents. Thus, synaptic currents
and the resulting post-synaptic potentials have a finite rise time.
The threshold crossing is followed by an absolute refractory period
during which the membrane potential is clamped to the resting potential.

The subthreshold membrane potential dynamics are given by [1]

dV_m/dt = - ( V_m - E_L ) / tau_m + I_syn(t) / C_m + I_e / C_m

where I_syn(t) is the sum of exp-shaped synaptic currents.


Parameters:
V_reset_( -56.0 - E_L ) // mV, reset potential, relative to E_L
t_ref_( 2.0 )     // ms, membrane time constant
g ( 12.5 )        // membrane conductance, in nS, = C/tau_m
g_rr ( 25.0 )     // auxiliary variable conductance, in nS, = C/tau_v
C_m( 250.0 )      // pF, membrane capacity
E_L( -70.0 )      // mV, leaking potential
tau_1( 10.0 )     // ms, time constant for the auxiliary variable
V_th( -50.0 - E_L ) // mV, spiking threshold, relative to E_L
tau_syn_ex( 2.0 ) // ms, time constant of excitatory synapses
tau_syn_in( 2.0 ) // ms, time constant of inhibitory synapses
I_e( 0.0 )        // pA, the external stimulating current
gsl_error_tol( 1e-6 ) // gsl throws if the tolerance is crossed

Remarks:
The original paper uses direct current input to elicit spiking.
The subthreshold behaviour is studied further in [2].

References:
[1] Richardson, M. J. E., Brunel, N. & Hakim, V.
    From subthreshold to firing-rate resonance.
    J. Neurophysiol. 89, 2538–2554 (2003).
[2] Brunel, N., Hakim, V. & Richardson, M. J. E.
    Firing-rate resonance in a generalized integrate-and-fire neuron with subthreshold resonance.
    Phys. Rev. E 67, 051916 (2003).

Sends: SpikeEvent

Receives: SpikeEvent, CurrentEvent, DataLoggingRequest

Author:  Mingers, May 2016.
Adapted from iaf_psc_alpha (September 1999, Diesmann, Gewaltig)
SeeAlso: iaf_psc_alpha, testsuite::test_iaf
*/

/**
 * Generalised leaky integrate-and-fire neuron (with 2 variables).
 */
namespace nest
{
/**
 * Function computing right-hand side of ODE for GSL solver.
 * @note Must be declared here so we can befriend it in class.
 * @note Must have C-linkage for passing to GSL. Internally, it is
 *       a first-class C++ function, but cannot be a member function
 *       because of the C-linkage.
 * @note No point in declaring it inline, since it is called
 *       through a function pointer.
 * @param void* Pointer to model neuron instance.
 */
extern "C" int gif2_psc_exp_dynamics( double, const double*, double*, void* );

class gif2_psc_exp : public Archiving_Node
{

public:
  gif2_psc_exp();
  gif2_psc_exp( const gif2_psc_exp& );
  ~gif2_psc_exp();

  /**
   * Import sets of overloaded virtual functions.
   * @see Technical Issues / Virtual Functions: Overriding, Overloading, and
   * Hiding
   */
  using Node::handle;
  using Node::handles_test_event;

  port send_test_event( Node&, rport, synindex, bool );

  void handle( SpikeEvent& );
  void handle( CurrentEvent& );
  void handle( DataLoggingRequest& );

  port handles_test_event( SpikeEvent&, rport );
  port handles_test_event( CurrentEvent&, rport );
  port handles_test_event( DataLoggingRequest&, rport );

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

private:
  void init_state_( const Node& proto );
  void init_buffers_();
  void calibrate();
  void update( const Time&, const long, const long );

  // END Boilerplate function declarations ----------------------------

  // Friends --------------------------------------------------------

  // make dynamics function quasi-member
  friend int gif2_psc_exp_dynamics( double, const double*, double*, void* );

  // The next two classes need to be friends to access the State_ class/member
  friend class RecordablesMap< gif2_psc_exp >;
  friend class UniversalDataLogger< gif2_psc_exp >;

  // private:
  // ----------------------------------------------------------------

  //! Independent parameters
  struct Parameters_
  {
    double_t V_reset_; //!< Reset Potential in mV
    double_t t_ref_;   //!< Refractory period in ms
    double_t g;     //!< Leak Conductance in nS, V_m
    double_t g_rr;     //!< Leak Conductance in nS, W, in the paper: g1
    double_t C_m;     //!< Membrane Capacitance in pF
    double_t E_L;     //!< Leak reversal Potential (aka resting potential) in mV
    double_t tau_1;   //!< adaptation time-constant in ms.
    double_t V_th;    //!< Spike threshold in mV.
    double_t tau_syn_ex; //!< Excitatory synaptic rise time.
    double_t tau_syn_in; //!< Excitatory synaptic rise time.
    double_t I_e;        //!< Intrinsic current in pA.
    double_t gsl_error_tol; //!< error bound for GSL integrator

    Parameters_(); //!< Sets default parameter values

    void get( DictionaryDatum& ) const; //!< Store current values in dictionary
    double set( const DictionaryDatum& ); //!< Set values from dicitonary
  };

public:
  // ----------------------------------------------------------------

  /**
   * State variables of the model.
   * @note Copy constructor and assignment operator required because
   *       of C-style array.
   */
  struct State_
  {
    /**
     * Enumeration identifying elements in state array State_::y_.
     * The state vector must be passed to GSL as a C array. This enum
     * identifies the elements of the vector. It must be public to be
     * accessible from the iteration function.
     */
    enum StateVecElems
    {
      V_M = 0,
      W,        // 1
      I_syn_ex, // 2
      I_syn_in, // 3
      STATE_VEC_SIZE
    };

    //! neuron state, must be C-array for GSL solver
    double_t y_[ STATE_VEC_SIZE ];
    int r_; //!< number of refractory steps remaining

    State_( const Parameters_& ); //!< Default initialization
    State_( const State_& );
    State_& operator=( const State_& );

    void get( DictionaryDatum&, const Parameters_& ) const;
    void set( const DictionaryDatum&, const Parameters_&, const double );
  };

  // ----------------------------------------------------------------

  /**
   * Buffers of the model.
   */
  struct Buffers_
  {
    Buffers_( gif2_psc_exp& );                  //!<Sets buffer pointers to 0
    Buffers_( const Buffers_&, gif2_psc_exp& ); //!<Sets buffer pointers to 0

    //! Logger for all analog data
    UniversalDataLogger< gif2_psc_exp > logger_;

    /** buffers and sums up incoming spikes/currents */
    RingBuffer spike_exc_;
    RingBuffer spike_inh_;
    RingBuffer currents_;

    /** GSL ODE stuff */
    gsl_odeiv_step* s_;    //!< stepping function
    gsl_odeiv_control* c_; //!< adaptive stepsize control function
    gsl_odeiv_evolve* e_;  //!< evolution function
    gsl_odeiv_system sys_; //!< struct describing system

    // IntergrationStep_ should be reset with the neuron on ResetNetwork,
    // but remain unchanged during calibration. Since it is initialized with
    // step_, and the resolution cannot change after nodes have been created,
    // it is safe to place both here.
    double_t step_;          //!< step size in ms
    double IntegrationStep_; //!< current integration time step, updated by GSL

    /**
     * Input current injected by CurrentEvent.
     * This variable is used to transport the current applied into the
     * _dynamics function computing the derivative of the state vector.
     * It must be a part of Buffers_, since it is initialized once before
     * the first simulation, but not modified before later Simulate calls.
     */
    double_t I_stim_;
  };

  // ----------------------------------------------------------------

  /**
   * Internal variables of the model.
   */
  struct Variables_
  {
    int RefractoryCounts_;
  };

  // Access functions for UniversalDataLogger -------------------------------

  //! Read out state vector elements, used by UniversalDataLogger
  template < State_::StateVecElems elem >
  double_t
  get_y_elem_() const
  {
    return S_.y_[ elem ];
  }

  // ----------------------------------------------------------------

  Parameters_ P_;
  State_ S_;
  Variables_ V_;
  Buffers_ B_;

  //! Mapping of recordables names to access functions
  static RecordablesMap< gif2_psc_exp > recordablesMap_;
};

inline port
gif2_psc_exp::send_test_event( Node& target,
  rport receptor_type,
  synindex,
  bool )
{
  SpikeEvent e;
  e.set_sender( *this );

  return target.handles_test_event( e, receptor_type );
}

inline port
gif2_psc_exp::handles_test_event( SpikeEvent&, rport receptor_type )
{
  if ( receptor_type != 0 )
    throw UnknownReceptorType( receptor_type, get_name() );
  return 0;
}

inline port
gif2_psc_exp::handles_test_event( CurrentEvent&, rport receptor_type )
{
  if ( receptor_type != 0 )
    throw UnknownReceptorType( receptor_type, get_name() );
  return 0;
}

inline port
gif2_psc_exp::handles_test_event( DataLoggingRequest& dlr,
  rport receptor_type )
{
  if ( receptor_type != 0 )
    throw UnknownReceptorType( receptor_type, get_name() );
  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
gif2_psc_exp::get_status( DictionaryDatum& d ) const
{
  P_.get( d );
  S_.get( d, P_ );
  Archiving_Node::get_status( d );

  ( *d )[ names::recordables ] = recordablesMap_.get_list();
}

inline void
gif2_psc_exp::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  const double delta_EL = ptmp.set( d );  // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d, ptmp, delta_EL );   // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  Archiving_Node::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

} // namespace

#endif // HAVE_GSL
#endif // GIF2_PSC_EXP_H
