# Moritz Helias

import scipy
import scipy.integrate
import scipy.stats
import scipy.special
#from matplotlib.pylab import *  # for plot
from numpy import *
#from mpmath import *   # use exakt function for taylor approximation
import math
import sympy
from sympy import abc
# firing rate after Brunel & Hakim 1999
# only true for delta shaped PSCs
#


#
# stationary firing rate of neuron with synaptic lp filter
# of time constant tau_s
# driven by GWN(mu,sigma)
# from Fourcoud & Brunel 2002
#

def taylor_fb_mu(tau,tau_s,tau_r,V_th,V_r,mu,sigma):
    def function(mu):
        return nu0_fb(tau,tau_s,tau_r,V_th,V_r,mu,sigma)
    x_0 = 0.
    coeff = taylor(function,x_0,3)
    return coeff

def Phi(s,s0=-7., order=4) :
    if s < s0 :#and s > -9. :
        f = sqrt(pi/2.)*sympy.exp(abc.x**2/2.)*(1+sympy.erf(abc.x/sqrt(2)))
        g = f.series(x0=s0,n=order).removeO()
        return float(g.evalf(subs={abc.x:(s-s0)}))

    else :
        if isinf(exp(s**2/2.)) : ### Added this if-branch to catch the case where (1+scipy.special.erf(s)) is already 0
            return 0.
        else :
            return sqrt(pi/2.)*(exp(s**2/2.)*(1+scipy.special.erf(s/sqrt(2))))

 
def Phi_old(x):
    if isinf(exp(x**2/2.)) : ### Added this if-branch to catch the case where (1+scipy.special.erf(s)) is already 0
        return 0.
    else :
        return sqrt(pi/2.)*(exp(x**2/2)*(1+scipy.special.erf(x/sqrt(2))))

def Phi_prime_sigma(s,sigma, s0=-7., order=4):
    if s<s0 :
        f = -sqrt(pi/2.)*abc.x/sigma*(abc.x*sympy.exp(abc.x**2/2.)*(1+sympy.erf(abc.x/sqrt(2)))+sqrt(2)/sqrt(pi))
        g = f.series(x0=s0,n=order).removeO()
        return float(g.evalf(subs={abc.x:(s-s0)}))
    else :
        if isinf(exp(s**2/2.)) :
            print 'warning, inf encountered'
            return -sqrt(pi/2.)*s/sigma*(2./sqrt(pi))
        else:
            return -sqrt(pi/2.)*s/sigma*(s*exp(s**2/2.)*(1+scipy.special.erf(s/sqrt(2.)))+sqrt(2)/sqrt(pi))

def Phi_prime_mu(s,sigma, s0=-7., order=4):
    if s<s0 :
        f = -sqrt(pi/2.)*sqrt(2)/sigma*(abc.x*sympy.exp(abc.x**2/2.)*(1+sympy.erf(abc.x/sqrt(2)))+sqrt(2)/sqrt(pi))
        g = f.series(x0=s0,n=order).removeO()
        return float(g.evalf(subs={abc.x:(s-s0)}))
    else :
        if isinf(exp(s**2/2.)) :
            print 'warning, inf encountered'
            return -sqrt(pi/2.)*s/sigma*(2./sqrt(pi))
        else:
            return -sqrt(pi/2.)*sqrt(2)/sigma*(s*exp(s**2/2.)*(1+scipy.special.erf(s/sqrt(2)))+sqrt(2)/sqrt(pi))



def Phi_prime_old(s,sigma) :
    if isinf(exp(s**2/2.)) :
        print 'warning, inf encountered'
        return -sqrt(pi/2.)*s/sigma*(2./sqrt(pi))
    else:
        return -sqrt(pi/2.)*s/sigma*(s*exp(s**2/2.)*(1+scipy.special.erf(s/sqrt(2.)))+sqrt(2)/sqrt(pi))


def nu0_fb(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    
    alpha = sqrt(2)*abs(scipy.special.zetac(0.5)+1)

    # effective threshold    
    V_th1 = V_th + sigma*alpha/2.*sqrt(tau_s/tau_m)

    # effective reset    
    V_r1 = V_r + sigma*alpha/2.*sqrt(tau_s/tau_m)

    # use standard Siegert with modified threshold and reset
    return nu_0(tau_m, tau_r, V_th1, V_r1, mu, sigma)

def nu0_fb433(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma,switch_fb=-7.):

    alpha = sqrt(2.)*abs(scipy.special.zetac(0.5)+1)
    
    x_th = sqrt(2.)*(V_th - mu)/sigma
    x_r = sqrt(2.)*(V_r - mu)/sigma
   
    if x_r < switch_fb:
        return nu0_fb(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma)

    # HB: preventing overflow in exponent in Phi(s)
    if x_th > 20.0/sqrt(2.):
        result = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)
    else:
        integral = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)*tau_m
        result = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma) - sqrt(tau_s/tau_m)*alpha/(tau_m*sqrt(2))*(Phi(x_th)-Phi(x_r))*integral**2
    if math.isnan(result):
        print mu, sigma, x_th, x_r
    return result


def nu0_fb433_old(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):

    alpha = sqrt(2.)*abs(scipy.special.zetac(0.5)+1)
    
    x_th = sqrt(2.)*(V_th - mu)/sigma
    x_r = sqrt(2.)*(V_r - mu)/sigma

    integral = 1./(nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)*tau_m)
    #print integral
    return nu_0(tau_m, tau_r, V_th, V_r, mu, sigma) - sqrt(tau_s/tau_m)*alpha/(tau_m*sqrt(2))*(Phi_old(x_th)-Phi_old(x_r))/integral**2

def d_nu_d_mu_fb433(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    alpha = sqrt(2)*abs(scipy.special.zetac(0.5)+1)
    x_th = sqrt(2)*(V_th - mu)/sigma
    x_r = sqrt(2)*(V_r - mu)/sigma
    
    integral = 1./(nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)*tau_m)
    
    prefactor = sqrt(tau_s/tau_m)*alpha/(tau_m*sqrt(2))
    
    return d_nu_d_mu_numeric(tau_m, tau_r, V_th, V_r, mu, sigma)-prefactor*((Phi_prime_mu(x_th, sigma)-Phi_prime_mu(x_r, sigma))*integral+(2*sqrt(2)/sigma)*(Phi(x_th)-Phi(x_r))**2)/integral**3




def d_nu_d_sigma_fb433(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    alpha = sqrt(2)*abs(scipy.special.zetac(0.5)+1)
    x_th = sqrt(2)*(V_th - mu)/sigma
    x_r = sqrt(2)*(V_r - mu)/sigma
    
    integral = 1./(nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)*tau_m)
    
    prefactor = sqrt(tau_s/tau_m)*alpha/(tau_m*sqrt(2))
    
    return d_nu_d_sigma_numeric(tau_m, tau_r, V_th, V_r, mu, sigma)-prefactor*((Phi_prime_sigma(x_th, sigma)-Phi_prime_sigma(x_r, sigma))*integral+(2./sigma)*(Phi(x_th)-Phi(x_r))*(x_th*Phi(x_th)-x_r*Phi(x_r)))/integral**3
    

    


# derivative of firing rate curve w.r.t mu
# for l.p. filtered synapses with tau_s
# effective threshold and reset from Fourcoud & Brunel 2002
#
def d_nu_d_mu_fb(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    alpha = sqrt(2)*abs(scipy.special.zetac(0.5)+1)

    # effective threshold    
    V_th1 = V_th + sigma*alpha/2.*sqrt(tau_s/tau_m)

    # effective reset    
    V_r1 = V_r + sigma*alpha/2.*sqrt(tau_s/tau_m)
    return d_nu_d_mu(tau_m, tau_r, V_th1, V_r1, mu, sigma)

def d_nu_d_mu_fb_numeric(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    alpha = sqrt(2)*abs(scipy.special.zetac(0.5)+1)

    # effective threshold    
    V_th1 = V_th + sigma*alpha/2.*sqrt(tau_s/tau_m)

    # effective reset    
    V_r1 = V_r + sigma*alpha/2.*sqrt(tau_s/tau_m)
    return d_nu_d_mu_numeric(tau_m, tau_r, V_th1, V_r1, mu, sigma)

def d2_nu_d_mu_fb_numeric(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    eps = 0.01
    nu0_minus = d_nu_d_mu_fb_numeric(tau_m, tau_s,tau_r, V_th, V_r, mu, sigma)
    nu0_plus = d_nu_d_mu_fb_numeric(tau_m, tau_s,tau_r, V_th, V_r, mu+eps, sigma)
    
    return (nu0_plus-nu0_minus)/eps




def d_nu_d_sigma_fb(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    alpha = sqrt(2)*abs(scipy.special.zetac(0.5)+1)

    # effective threshold    
    V_th1 = V_th + sigma*alpha/2.*sqrt(tau_s/tau_m)

    # effective reset    
    V_r1 = V_r + sigma*alpha/2.*sqrt(tau_s/tau_m)
    return d_nu_d_sigma(tau_m, tau_r, V_th1, V_r1, mu, sigma)


#
# derivative of nu_0 by input rate
# for l.p. filtered synapses with tau_s
# effective threshold and reset from Fourcoud & Brunel 2002
#
def d_nu_d_nu_in_fb(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma, w):

    alpha = sqrt(2)*abs(scipy.special.zetac(0.5)+1)

    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma

    y_th_fb = y_th + alpha/2.*sqrt(tau_s/tau_m)
    y_r_fb = y_r + alpha/2.*sqrt(tau_s/tau_m)

    nu0 = nu0_fb(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma)


    # linear contribution
    lin = sqrt(pi) * (tau_m*nu0)**2 * w/sigma * (exp(y_th_fb**2) * (1 + scipy.special.erf(y_th_fb)) - exp(y_r_fb**2) * (1 + scipy.special.erf(y_r_fb)) )

    # quadratic contribution
    sqr = sqrt(pi) * (tau_m*nu0)**2 * w/sigma * (exp(y_th_fb**2) * (1 + scipy.special.erf(y_th_fb)) * 0.5*y_th*w/sigma - exp(y_r_fb**2) * (1 + scipy.special.erf(y_r_fb)) * 0.5*y_r*w/sigma )

    return lin + sqr, lin, sqr


#
# for mu < V_th
#
def siegert1(tau_m, tau_r, V_th, V_r, mu, sigma):
    #print 'using siegert1'
    y_th = (V_th - mu)/sigma
    #print 'y_th: ' + str(y_th)
    y_r = (V_r - mu)/sigma
   
    def integrand(u):
        if u == 0:
            return exp(-y_th**2)*2*(y_th - y_r)
        else:
            return exp(-(u-y_th)**2) * ( 1.0 - exp(2*(y_r-y_th)*u) ) / u

    lower_bound = y_th
    err_dn = 1.0
    while err_dn > 1e-12 and lower_bound > 1e-16:
        err_dn = integrand(lower_bound)
        if err_dn > 1e-12:            
            lower_bound /= 2

    upper_bound = y_th
    err_up = 1.0
    while err_up > 1e-12:
       err_up = integrand(upper_bound)
       if err_up > 1e-12:
           upper_bound *= 2

    err = max(err_up, err_dn)

    #print 'upper_bound = ', upper_bound
    #print 'lower_bound = ', lower_bound
    #print 'err_dn = ', err_dn
    #print 'err_up = ', err_up

    # SA, added check to prevent overflow:
    if y_th>=20:
        out = 0.
    if y_th<20:
        out = 1.0/(tau_r + exp(y_th**2)*scipy.integrate.quad(integrand, lower_bound, upper_bound)[0] * tau_m)
    else:                                                                                                                     # added since y_th can be NAN, for example when siegert is called by fsolve  
        out = 0.
        
    return out
 
#
# for mu > V_th
#
def siegert2(tau_m, tau_r, V_th, V_r, mu, sigma):
    #print 'using siegert2'
    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma

    def integrand(u):
        if u == 0:
            return 2*(y_th - y_r)
        else:
            return ( exp(2*y_th*u -u**2) - exp(2*y_r*u -u**2) ) / u

    upper_bound = 1.0
    err = 1.0
    while err > 1e-12:
        err = integrand(upper_bound)
        upper_bound *= 2

    return 1.0/(tau_r + scipy.integrate.quad(integrand, 0.0, upper_bound)[0] * tau_m)


   
    
def nu_0(tau_m, tau_r, V_th, V_r, mu, sigma):
    """tau_m: membrane time constant"""
    """tau_r: refractory time constant"""
    """V_th: threshold"""
    """V_r: reset potential"""
    """mu: mean input"""
    """sigma: std of equivalent GWN input"""

    if mu <= V_th+(0.95*abs(V_th)-abs(V_th)):
    #if  mu <= 0.95*V_th:
        return siegert1(tau_m, tau_r, V_th, V_r, mu, sigma)
    else:
        return siegert2(tau_m, tau_r, V_th, V_r, mu, sigma)


def nu_0_bc(tau_m, tau_r, V_th, V_r, mu, sigma,a,b):
    A = sigma/sqrt(2.)*a
    B = sigma/sqrt(2.)*b
        

    x_th = sqrt(2)*(V_th - mu)/sigma
    x_r = sqrt(2)*(V_r - mu)/sigma
    def F(x):
        return sqrt(pi/2.)*(scipy.special.erf(x/sqrt(2))+1)
    
    nom = 1-A*exp(x_th**2/2.)*F(x_th)+B*exp(x_r**2/2.)*F(x_r)
    return nom*nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)
    



#
# corrections of the siegert function for real weights and discretizations
#

###helper funtions ###
def nu_exc_inh_from_mu_sigma(w, g, tau_m, mu, sigma):
  '''si units please'''
  nu_exc_0 = mu / (w*tau_m)
  nu_bal = ((sigma**2 / w**2) - mu/w) / (1.0 + g) / tau_m
 
  nu_exc = nu_exc_0 + nu_bal
  nu_inh = nu_bal / g
  
  return (nu_exc, nu_inh)


def mu_sigma_from_nu_w_g(nu_e, nu_i, w, g, tau_m):
  '''si units please'''

  mu = tau_m * w * (nu_e - g*nu_i)
  sigma = w * sqrt(tau_m*(nu_e + g**2*nu_i))
  
  return (mu, sigma)



def nu_0_tom(tau_m, V_th, V_r, mu, sigma):

    def integrand(u):
        if u < -4.0:
            return -1/sqrt(pi) * ( 1.0/u - 1.0/(2.0*u**3) + 3.0/(4.0*u**5) - 15.0/(8.0*u**7) )
        else:
            return exp(u**2)*(1+scipy.special.erf(u))

    upper_bound = (V_th - mu)/sigma
    lower_bound = (V_r - mu)/sigma

    #dx = 1e-3
    #I = 0.0
    #for x in arange(lower_bound, upper_bound, dx):
    #    I += integrand(x)*dx
    #return 1.0 / (I * tau_m * sqrt(pi))    
    return 1.0 / ( scipy.integrate.quad(integrand, lower_bound, upper_bound)[0] * tau_m * sqrt(pi) )
    

#
# derivative of nu_0 by mu
#
def d_nu_d_mu(tau_m, tau_r, V_th, V_r, mu, sigma):
    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma
    
    nu0 = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)
    #if isinf(exp(y_r**2)) : ### To catch the case of very high mu where both exponentials should cancel
    #    return 0.0
    #else :
    return sqrt(pi) * tau_m * nu0**2 / sigma * (exp(y_th**2) * (1 + scipy.special.erf(y_th)) - exp(y_r**2) * (1 + scipy.special.erf(y_r)))


def d_nu_d_sigma_fb_numeric(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    alpha = sqrt(2)*abs(scipy.special.zetac(0.5)+1)

    # effective threshold    
    V_th1 = V_th + sigma*alpha/2.*sqrt(tau_s/tau_m)

    # effective reset    
    V_r1 = V_r + sigma*alpha/2.*sqrt(tau_s/tau_m)
    return d_nu_d_sigma_numeric(tau_m, tau_r, V_th1, V_r1, mu, sigma)


def d_nu_d_mu_numeric(tau_m, tau_r, V_th, V_r, mu, sigma):
    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma
    eps = 0.01
    nu0_minus = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)
    nu0_plus = nu_0(tau_m, tau_r, V_th, V_r, mu+eps, sigma)
    
    return (nu0_plus-nu0_minus)/eps

def d_nu_d_mu_fb433_numeric(tau_m, tau_s,tau_r, V_th, V_r, mu, sigma,switch_fb='none'):
    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma
    eps = 0.01
    nu0_minus = nu0_fb433(tau_m, tau_s,tau_r, V_th, V_r, mu, sigma,switch_fb)
    nu0_plus = nu0_fb433(tau_m, tau_s, tau_r, V_th, V_r, mu+eps, sigma,switch_fb)
    
    return (nu0_plus-nu0_minus)/eps

def d_nu_d_sigma_fb433_numeric(tau_m, tau_s ,tau_r, V_th, V_r, mu, sigma,switch_fb='none'):
    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma
    eps = 0.01
    nu0_minus = nu0_fb433(tau_m, tau_s,tau_r, V_th, V_r, mu, sigma,switch_fb)
    nu0_plus = nu0_fb433(tau_m, tau_s,tau_r, V_th, V_r, mu, sigma+eps,switch_fb)
    
    return (nu0_plus-nu0_minus)/eps



def d_nu_d_sigma_numeric(tau_m, tau_r, V_th, V_r, mu, sigma):
    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma
    eps = 0.01
    nu0_minus = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)
    nu0_plus = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma+eps)
    
    return (nu0_plus-nu0_minus)/eps



def d_nu_d_sigma(tau_m, tau_r, V_th, V_r, mu, sigma):

    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma

    nu0 = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)

    return sqrt(pi) * tau_m * nu0**2 / sigma * (exp(y_th**2) * (1 + scipy.special.erf(y_th))*y_th - exp(y_r**2) * (1 + scipy.special.erf(y_r))*y_r )




#
# derivative of nu_0 by input rate
#
def d_nu_d_nu_in(tau_m, tau_r, V_th, V_r, mu, sigma, w):

    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma

    nu0 = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)

    return sqrt(pi) * (tau_m*nu0)**2 * w/sigma * (exp(y_th**2) * (1 + scipy.special.erf(y_th)) * (1. + 0.5*y_th*w/sigma) - exp(y_r**2) * (1 + scipy.special.erf(y_r)) * (1. + 0.5*y_r*w/sigma) )

#
# membrane potential distribution
# according to iaf_foker_planck.lyx
#
def P(nu0, tau_m, V_th, V_r, mu, sigma, V):

    y = (V-mu)/sigma
    y_th = (V_th-mu)/sigma
    y_r = (V_r-mu)/sigma

    def integrand(u):
        return exp(u**2)

    if (y < y_r):
        I = scipy.integrate.quad(integrand, y_r, y_th)[0]
    else:
        I = scipy.integrate.quad(integrand, y, y_th)[0]

       
    return 2.0*nu0*tau_m/sigma * I * exp(-y**2)

def P_fb(nu0, tau_m,tau_s, V_th, V_r, mu, sigma, V):

    y = (V-mu)/sigma
    y_th = (V_th-mu)/sigma
    y_r = (V_r-mu)/sigma
    alpha = sqrt(2)*abs(scipy.special.zetac(0.5)+1)

    # effective threshold    
    y_r1 = y_r + alpha/2.*sqrt(tau_s/tau_m)

    # effective reset    
    y_th1 = y_th + alpha/2.*sqrt(tau_s/tau_m) 
    y_th1_r = y_th + alpha/2.*sqrt(tau_s/tau_m/2.) 
    

    def integrand(u):
        return exp(u**2)

    if (y < y_r):
        I = scipy.integrate.quad(integrand, y_r1, y_th1)[0]
    else:
        I = scipy.integrate.quad(integrand, y, y_th1)[0]

       
    return 2.0*nu0*tau_m/sigma * I * exp(-y**2)


def P_bc(nu0,tau_m,V_th, V_r,a,b,mu,sigma,V):
    Phi_0 = nu0*tau_m
    A = sigma/sqrt(2.)*a
    B = sigma/sqrt(2.)*b
    x = sqrt(2)*(V-mu)/sigma
    x_th = sqrt(2)*(V_th-mu)/sigma
    x_r = sqrt(2)*(V_r-mu)/sigma
    def int_exp2(a,b):
        int_b = sqrt(pi/2.)*1j*scipy.special.erf(-1j*b/sqrt(2))
        int_a = sqrt(pi/2.)*1j*scipy.special.erf(-1j*a/sqrt(2))  
        return int_b - int_a
    
    def D(A,B,nu0,x_th,x_r):
        return A*exp(x_th**2/2.)-B*exp(x_r**2/2.)+Phi_0*(int_exp2(x_r,x_th))
    def C(A,x_th):
        return A*exp(x_th**2/2.)

    if x<=x_r:
        #print 'DD',D(A,B,nu0,y_th,y_r)
        return (D(A,B,nu0,x_th,x_r)*exp(-x**2/2.))*sqrt(2)/sigma
    if x>x_r:
        return (C(A,x_th)*exp(-x**2/2.)+Phi_0*exp(-x**2/2.)*(int_exp2(x,x_th)))*sqrt(2)/sigma



    # get the coefficient of variation squared for given input parameters
    # according to brunel00
    # assumes, that self.set_mu_sigma(mu, sigma) was called before
def CV2(nu_0, tau_m, tau_r, V_th, V_r, mu, sigma):

        y_th = (V_th - mu)/sigma
        y_r = (V_r - mu)/sigma

        def integrand_outer(s):            

            def integrand_inner(t):
                if t > 0.0:
                    return 1.0/t * (exp(-s**2 - 2*t**2 + 2*s*t) - exp(-s**2))
                else:
                    return 2*s

            return 1.0/s * (exp(2*y_th*s) - exp(2*y_r*s)) * scipy.integrate.quad(integrand_inner, 0.0, s)[0]

        s_up = 1.0
        err = 1.0
        while err > 1e-12:
            s_up *= 2
            err = integrand_outer(s_up)

        print 'upper bound=', s_up
        print 'integrand=', err

        return 2.0 * (nu_0*tau_m)**2 * scipy.integrate.quad(integrand_outer, 0.0, s_up)[0]

