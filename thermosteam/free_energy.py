# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Free energy functors for Chemical objects.

"""
from .constants import R
from .base import functor, PhaseTFunctorBuilder, PhaseTPFunctorBuilder
from math import log
from scipy.integrate import quad

epsrel=1e-5  #Relative error for integral solver

def get_excess_energy(eos, T, P, free_energy, phase):
    eos = eos.to(T, P)
    name = f"{free_energy}_dep_{phase}"
    try: return getattr(eos, name)
    except: # Maybe identified closer to another phase near supercritical conditions (doesn't matter)
        name = f"{free_energy}_dep_g" if phase == 'l' else f"{free_energy}_dep_l"
        return getattr(eos, name)
@functor(var='H')
def Enthalpy(T, P, Cn, T_ref, H_ref,P_ref=101325,V=None,phase=None):
    term1=H_ref + Cn.T_dependent_property_integral(T_ref, T)
    if V is not None and P!=P_ref:
        if phase in ['l','L','s','S']: #Incompressible
            term2=V(T=T,P=P)*(P-P_ref) #m^3/mol * dP giving J/mol or kJ/kmol
        else:
            term2, _ = quad(pressure_term, P_ref,P, args=(V, T),epsrel=epsrel) #m^3/mol * dP giving J/mol or kJ/kmol
        return term1+term2
    else:
        return term1

# @functor(var='H')
# def Enthalpy(T, P, Cn, T_ref, H_ref):
#     return H_ref + Cn.T_dependent_property_integral(T_ref, T)

def pressure_term(P,V,T): #For quad, first variable should be independent variable
    return V(T=T,P=P)-(T*V.TP_dependent_property_derivative_T(T,P)) # units are in m^3/mol, for derivative see thermo package utils/tp_dependent_property.py/TPDependentProperty/TP_dependent_property_derivative_T
def pressure_term_entropy(P,V,T):
    return V.TP_dependent_property_derivative_T(T,P)
@functor(var='S')
def Entropy(T, P, Cn, T_ref, S0):
    return S0 + Cn.T_dependent_property_integral_over_T(T_ref, T)

@functor(var='S')
def EntropyGas(T, P, Cn, T_ref, P_ref, S0,V=None,ideal=True):
    term1=S0 + Cn.T_dependent_property_integral_over_T(T_ref, T)
    if ideal:
        term2=R*log(P/P_ref)
    else:
        term2, _= quad(pressure_term_entropy, P_ref,P, args=(V, T),epsrel=epsrel)
    return term1-term2

# @functor(var='S')
# def EntropyGas(T, P, Cn, T_ref, P_ref, S0):
#     return S0 + Cn.T_dependent_property_integral_over_T(T_ref, T) - R*log(P/P_ref)

@functor(var='H.l')
def Liquid_Enthalpy_Ref_Liquid(T, P, Cn_l, T_ref, H_ref,P_ref=101325,V_l=None): #See thermo package chemical.py/Chemical
    """Enthapy (kJ/kmol)  assuming the specified phase."""
    term1=H_ref + Cn_l.T_dependent_property_integral(T_ref, T)
    if V_l is not None and P!=P_ref: #Pressure changes
        term2=V_l(T=T,P=P)*(P-P_ref) #m^3/mol * dP giving J/mol or kJ/kmol
        # term2, _ = quad(pressure_term, P_ref,P, args=(V_l, T),epsrel=epsrel) #m^3/mol * dP giving J/mol or kJ/kmol
        return term1+term2
    else: 
        return term1

# @functor(var='H.l')
# def Liquid_Enthalpy_Ref_Liquid(T, P, Cn_l, T_ref, H_ref):
#     """Enthapy (kJ/kmol) disregarding pressure and assuming the specified phase."""
#     return H_ref + Cn_l.T_dependent_property_integral(T_ref, T)

@functor(var='H.l')
def Liquid_Enthalpy_Ref_Gas(T, P, Cn_l, H_int_Tb_to_T_ref_g, Hvap_Tb, Tb, H_ref,P_ref=101325,V_l=None):
    term1 = H_ref - H_int_Tb_to_T_ref_g - Hvap_Tb + Cn_l.T_dependent_property_integral(Tb, T)
    if V_l is not None and P!=P_ref: #Pressure changes
        term2=V_l(T=T,P=P)*(P-P_ref) #m^3/mol * dP giving J/mol or kJ/kmol
        # term2, _ = quad(pressure_term, P_ref,P, args=(V_l, T),epsrel=epsrel) #m^3/mol * dP giving J/mol or kJ/kmol
        return term1+term2
    else:
        return term1
    

# @functor(var='H.l')
# def Liquid_Enthalpy_Ref_Gas(T, P, Cn_l, H_int_Tb_to_T_ref_g, Hvap_Tb, Tb, H_ref):
#     return H_ref - H_int_Tb_to_T_ref_g - Hvap_Tb + Cn_l.T_dependent_property_integral(Tb, T)

@functor(var='H.l')
def Liquid_Enthalpy_Ref_Solid(T, P, Cn_l, H_int_T_ref_to_Tm_s, Hfus, Tm, H_ref, P_ref=101325,V_l=None):
    term1=H_ref + H_int_T_ref_to_Tm_s + Hfus + Cn_l.T_dependent_property_integral(Tm, T)
    if V_l is not None and P!=P_ref: #Pressure changes
        term2=V_l(T=T,P=P)*(P-P_ref) #m^3/mol * dP giving J/mol or kJ/kmol
        # term2, _ = quad(pressure_term, P_ref,P, args=(V_l, T),epsrel=epsrel) #m^3/mol * dP giving J/mol or kJ/kmol
        return term1+term2
    else:
        return term1
    
# @functor(var='H.l')
# def Liquid_Enthalpy_Ref_Solid(T, Cn_l, H_int_T_ref_to_Tm_s, Hfus, Tm, H_ref):
#     return H_ref + H_int_T_ref_to_Tm_s + Hfus + Cn_l.T_dependent_property_integral(Tm, T)

@functor(var='H.s')
def Solid_Enthalpy_Ref_Solid(T, P, Cn_s, T_ref, H_ref, P_ref=101325,V_s=None):
    term1=H_ref + Cn_s.T_dependent_property_integral(T_ref, T)
    if V_s is not None and P!=P_ref: #Pressure changes
        term2=V_s(T=T,P=P)*(P-P_ref)
        return term1+term2
    else:
        return term1

# @functor(var='H.s')
# def Solid_Enthalpy_Ref_Solid(T, P, Cn_s, T_ref, H_ref):
#     return H_ref + Cn_s.T_dependent_property_integral(T_ref, T)

@functor(var='H.s')
def Solid_Enthalpy_Ref_Liquid(T, P, Cn_s, H_int_Tm_to_T_ref_l, Hfus, Tm, H_ref,P_ref=101325,V_s=None):  
    term1=H_ref - H_int_Tm_to_T_ref_l - Hfus + Cn_s.T_dependent_property_integral(Tm, T)
    if V_s is not None and P!=P_ref:
        term2=V_s(T=T,P=P)*(P-P_ref)
        return term1+term2
    else:
        return term1

# @functor(var='H.s')
# def Solid_Enthalpy_Ref_Liquid(T, P, Cn_s, H_int_Tm_to_T_ref_l, Hfus, Tm, H_ref):
#     return H_ref - H_int_Tm_to_T_ref_l - Hfus + Cn_s.T_dependent_property_integral(Tm, T)

@functor(var='H.s')
def Solid_Enthalpy_Ref_Gas(T, P, Cn_s, H_int_Tb_to_T_ref_g, Hvap_Tb,
                           H_int_Tm_to_Tb_l, Hfus, Tm, H_ref, P_ref=101325,V_s=None):
    term1=H_ref - H_int_Tb_to_T_ref_g - Hvap_Tb - H_int_Tm_to_Tb_l - Hfus + Cn_s.T_dependent_property_integral(Tm, T)
    if V_s is not None and P!=P_ref:
        term2=V_s(T=T,P=P)*(P-P_ref)
        return term1+term2
    else:
        return term1

# @functor(var='H.s')
# def Solid_Enthalpy_Ref_Gas(T, P, Cn_s, H_int_Tb_to_T_ref_g, Hvap_Tb,
#                            H_int_Tm_to_Tb_l, Hfus, Tm, H_ref):
#     return H_ref - H_int_Tb_to_T_ref_g - Hvap_Tb - H_int_Tm_to_Tb_l - Hfus + Cn_s.T_dependent_property_integral(Tm, T)

@functor(var='H.g')
def Gas_Enthalpy_Ref_Gas(T, P, Cn_g, T_ref, H_ref, P_ref=101325,V_g=None):
    term1=H_ref + Cn_g.T_dependent_property_integral(T_ref, T)
    if V_g is not None and P!=P_ref:
        term2, _ = quad(pressure_term, P_ref,P, args=(V_g, T),epsrel=epsrel) #m^3/mol * dP giving J/mol or kJ/kmol
        return term1+term2
    else:
        return term1
        
    
# @functor(var='H.g')
# def Gas_Enthalpy_Ref_Gas(T, P, Cn_g, T_ref, H_ref):
#     return H_ref + Cn_g.T_dependent_property_integral(T_ref, T)

@functor(var='H.g')
def Gas_Enthalpy_Ref_Liquid(T, P, Cn_g, H_int_T_ref_to_Tb_l, Hvap_Tb, 
                            Tb, H_ref,P_ref=101325,V_g=None):
    term1=H_ref + H_int_T_ref_to_Tb_l + Hvap_Tb + Cn_g.T_dependent_property_integral(Tb, T)
    if V_g is not None and P!=P_ref:
        term2, _ = quad(pressure_term, P_ref,P, args=(V_g, T),epsrel=epsrel) #m^3/mol * dP giving J/mol or kJ/kmol
        return term1+term2
    else:
        return term1

# @functor(var='H.g')
# def Gas_Enthalpy_Ref_Liquid(T, P, Cn_g, H_int_T_ref_to_Tb_l, Hvap_Tb, 
#                             Tb, H_ref):
#     return H_ref + H_int_T_ref_to_Tb_l + Hvap_Tb + Cn_g.T_dependent_property_integral(Tb, T)

@functor(var='H.g')
def Gas_Enthalpy_Ref_Solid(T, P, Cn_g, H_int_T_ref_to_Tm_s, Hfus,
                           H_int_Tm_to_Tb_l, Hvap_Tb, Tb, H_ref, P_ref=101325,V_g=None):
    term1=H_ref + H_int_T_ref_to_Tm_s + Hfus + H_int_Tm_to_Tb_l + Hvap_Tb + Cn_g.T_dependent_property_integral(Tb, T)
    if V_g is not None and P!=P_ref:
        term2, _ = quad(pressure_term, P_ref,P, args=(V_g, T),epsrel=epsrel) #m^3/mol * dP giving J/mol or kJ/kmol
        return term1+term2
    else:
        return term1

# @functor(var='H.g')
# def Gas_Enthalpy_Ref_Solid(T, P, Cn_g, H_int_T_ref_to_Tm_s, Hfus,
#                            H_int_Tm_to_Tb_l, Hvap_Tb, Tb, H_ref):
#     return H_ref + H_int_T_ref_to_Tm_s + Hfus + H_int_Tm_to_Tb_l + Hvap_Tb + Cn_g.T_dependent_property_integral(Tb, T)


EnthalpyRefLiquid = PhaseTPFunctorBuilder('H',
                                        Solid_Enthalpy_Ref_Liquid.functor,
                                        Liquid_Enthalpy_Ref_Liquid.functor,
                                        Gas_Enthalpy_Ref_Liquid.functor)

EnthalpyRefSolid = PhaseTPFunctorBuilder('H',
                                       Solid_Enthalpy_Ref_Solid.functor,
                                       Liquid_Enthalpy_Ref_Solid.functor,
                                       Gas_Enthalpy_Ref_Solid.functor)

EnthalpyRefGas = PhaseTPFunctorBuilder('H',
                                     Solid_Enthalpy_Ref_Gas.functor,
                                     Liquid_Enthalpy_Ref_Gas.functor,
                                     Gas_Enthalpy_Ref_Gas.functor)

@functor(var='S.l')
def Liquid_Entropy_Ref_Liquid(T, P, Cn_l, T_ref, S0):
    """Enthapy (kJ/kmol) disregarding pressure and assuming the specified phase."""
    return S0 + Cn_l.T_dependent_property_integral_over_T(T_ref, T)

@functor(var='S.l')
def Liquid_Entropy_Ref_Gas(T, P, Cn_l, S_int_Tb_to_T_ref_g, Svap_Tb, Tb, S0):
    return S0 - S_int_Tb_to_T_ref_g - Svap_Tb + Cn_l.T_dependent_property_integral_over_T(Tb, T)
    
@functor(var='S.l')
def Liquid_Entropy_Ref_Solid(T, P, Cn_l, S_int_T_ref_to_Tm_s, Sfus, Tm, S0):
    return S0 + S_int_T_ref_to_Tm_s + Sfus + Cn_l.T_dependent_property_integral_over_T(Tm, T)
    
@functor(var='S.s')
def Solid_Entropy_Ref_Solid(T, P, Cn_s, T_ref, S0):
    return S0 + Cn_s.T_dependent_property_integral_over_T(T_ref, T)

@functor(var='S.s')
def Solid_Entropy_Ref_Liquid(T, P, Cn_s, S_int_Tm_to_T_ref_l, Sfus, Tm, S0):
    return S0 - S_int_Tm_to_T_ref_l - Sfus + Cn_s.T_dependent_property_integral_over_T(Tm, T)

@functor(var='S.s')
def Solid_Entropy_Ref_Gas(T, P, Cn_s, S_int_Tb_to_T_ref_g, Svap_Tb, 
                          S_int_Tm_to_Tb_l, Sfus, Tm, S0):
    return S0 - S_int_Tb_to_T_ref_g - Svap_Tb - S_int_Tm_to_Tb_l - Sfus + Cn_s.T_dependent_property_integral_over_T(Tm, T)

@functor(var='S.g')
def Gas_Entropy_Ref_Gas(T, P, Cn_g, T_ref, P_ref, S0,V_g=None,ideal=True):
    term1=S0 + Cn_g.T_dependent_property_integral_over_T(T_ref, T)
    if ideal:
        term2=R*log(P/P_ref)
    else:
        term2, _= quad(pressure_term_entropy, P_ref,P, args=(V_g, T),epsrel=epsrel)
    return term1-term2
   
# @functor(var='S.g')
# def Gas_Entropy_Ref_Gas(T, P, Cn_g, T_ref, P_ref, S0):
#     return S0 + Cn_g.T_dependent_property_integral_over_T(T_ref, T) - R*log(P/P_ref)

@functor(var='S.g')
def Gas_Entropy_Ref_Liquid(T, P, Cn_g, S_int_T_ref_to_Tb_l, Svap_Tb,
                           Tb, P_ref, S0, V_g=None,ideal=True):
    term1=S0 + S_int_T_ref_to_Tb_l + Svap_Tb + Cn_g.T_dependent_property_integral_over_T(Tb, T)
    if ideal:
        term2=R*log(P/P_ref)
    else:
        term2, _= quad(pressure_term_entropy, P_ref,P, args=(V_g, T),epsrel=epsrel)
    return term1-term2
# @functor(var='S.g')
# def Gas_Entropy_Ref_Liquid(T, P, Cn_g, S_int_T_ref_to_Tb_l, Svap_Tb,
#                            Tb, P_ref, S0):
#     return S0 + S_int_T_ref_to_Tb_l + Svap_Tb + Cn_g.T_dependent_property_integral_over_T(Tb, T) - R*log(P/P_ref)

@functor(var='S.g')
def Gas_Entropy_Ref_Solid(T, P, Cn_g, S_int_T_ref_to_Tm_s, Sfus,
                          S_int_Tm_to_Tb_l, Svap_Tb, Tb, P_ref, S0, V_g=None,ideal=True):
    term1=S0 + S_int_T_ref_to_Tm_s + Sfus + S_int_Tm_to_Tb_l + Svap_Tb + Cn_g.T_dependent_property_integral_over_T(Tb, T)
    if ideal:
        term2=R*log(P/P_ref)
    else:
        term2, _= quad(pressure_term_entropy, P_ref,P, args=(V_g, T),epsrel=epsrel)
    return term1-term2


EntropyRefLiquid = PhaseTPFunctorBuilder('S',
                                        Solid_Entropy_Ref_Liquid.functor,
                                        Liquid_Entropy_Ref_Liquid.functor,
                                        Gas_Entropy_Ref_Liquid.functor)

EntropyRefSolid = PhaseTPFunctorBuilder('S',
                                       Solid_Entropy_Ref_Solid.functor,
                                       Liquid_Entropy_Ref_Solid.functor,
                                       Gas_Entropy_Ref_Solid.functor)

EntropyRefGas = PhaseTPFunctorBuilder('S',
                                     Solid_Entropy_Ref_Gas.functor,
                                     Liquid_Entropy_Ref_Gas.functor,
                                     Gas_Entropy_Ref_Gas.functor)

@functor(var='H.l')
def Excess_Liquid_Enthalpy_Ref_Liquid(T, P):
    return 0

@functor(var='H.l')
def Excess_Liquid_Enthalpy_Ref_Gas(T, P, eos, H_dep_Tb_Pb_g,
                                   H_dep_Tb_P_ref_g):
    if P != 101325:
        dH_dep_l = get_excess_energy(eos, T, P, 'H', 'l') - get_excess_energy(eos, T, 101325, 'H', 'l')
    else:
        dH_dep_l = 0.
    return H_dep_Tb_Pb_g - H_dep_Tb_P_ref_g + dH_dep_l
    
@functor(var='H.l')
def Excess_Liquid_Enthalpy_Ref_Solid(T, P):
    return 0
    
@functor(var='H.s')
def Excess_Solid_Enthalpy_Ref_Solid(T, P):
    return 0

@functor(var='H.s')
def Excess_Solid_Enthalpy_Ref_Liquid(T, P):
    return 0

@functor(var='H.s')
def Excess_Solid_Enthalpy_Ref_Gas(T, P):
    return 0
    
@functor(var='H.g')
def Excess_Gas_Enthalpy_Ref_Gas(T, P, eos, H_dep_ref_g):
    H_dep_g = get_excess_energy(eos, T, P, 'H', 'g')
    return H_dep_g - H_dep_ref_g

@functor(var='H.g')
def Excess_Gas_Enthalpy_Ref_Liquid(T, P, eos, H_dep_T_ref_Pb,
                                   H_dep_ref_l, H_dep_Tb_Pb_g):
    H_dep_g = get_excess_energy(eos, T, P, 'H', 'g')
    return H_dep_T_ref_Pb - H_dep_ref_l + H_dep_g - H_dep_Tb_Pb_g

@functor(var='H.g')
def Excess_Gas_Enthalpy_Ref_Solid(T):
    return 0

ExcessEnthalpyRefLiquid = PhaseTPFunctorBuilder('H_excess',
                                               Excess_Solid_Enthalpy_Ref_Liquid.functor,
                                               Excess_Liquid_Enthalpy_Ref_Liquid.functor,
                                               Excess_Gas_Enthalpy_Ref_Liquid.functor)

ExcessEnthalpyRefSolid = PhaseTPFunctorBuilder('H_excess',
                                                Excess_Solid_Enthalpy_Ref_Solid.functor,
                                                Excess_Liquid_Enthalpy_Ref_Solid.functor,
                                                Excess_Gas_Enthalpy_Ref_Solid.functor)

ExcessEnthalpyRefGas = PhaseTPFunctorBuilder('H_excess',
                                            Excess_Solid_Enthalpy_Ref_Gas.functor,
                                            Excess_Liquid_Enthalpy_Ref_Gas.functor,
                                            Excess_Gas_Enthalpy_Ref_Gas.functor)

@functor(var='S.l')
def Excess_Liquid_Entropy_Ref_Liquid(T, P):
    return 0

@functor(var='S.l')
def Excess_Liquid_Entropy_Ref_Gas(T, P, eos, S_dep_Tb_Pb_g,
                                  S_dep_Tb_P_ref_g):
    if P != 101325:
        dS_dep_l = get_excess_energy(eos, T, P, 'S', 'l') - get_excess_energy(eos, T, 101325, 'S', 'l')
    else:
        dS_dep_l = 0.
    return S_dep_Tb_Pb_g - S_dep_Tb_P_ref_g + dS_dep_l
    
@functor(var='S.l')
def Excess_Liquid_Entropy_Ref_Solid(T, P):
    return 0
    
@functor(var='S.s')
def Excess_Solid_Entropy_Ref_Solid(T, P):
    return 0

@functor(var='S.s')
def Excess_Solid_Entropy_Ref_Liquid(T, P):
    return 0

@functor(var='S.s')
def Excess_Solid_Entropy_Ref_Gas(T, P):
    return 0
    
@functor(var='S.g')
def Excess_Gas_Entropy_Ref_Gas(T, P, eos, S_dep_ref_g):
    S_dep_g = get_excess_energy(eos, T, P, 'S', 'g')
    return S_dep_g - S_dep_ref_g

@functor(var='S.g')
def Excess_Gas_Entropy_Ref_Liquid(T, P, eos, S_dep_T_ref_Pb, 
                                  S_dep_ref_l, S_dep_Tb_Pb_g):
    S_dep_g = get_excess_energy(eos, T, P, 'S', 'g')
    return S_dep_T_ref_Pb - S_dep_ref_l + S_dep_g - S_dep_Tb_Pb_g

@functor(var='S.g')
def Excess_Gas_Entropy_Ref_Solid(T, P):
    return 0

ExcessEntropyRefLiquid = PhaseTPFunctorBuilder('S_excess',
                                              Excess_Solid_Entropy_Ref_Liquid.functor,
                                              Excess_Liquid_Entropy_Ref_Liquid.functor,
                                              Excess_Gas_Entropy_Ref_Liquid.functor)

ExcessEntropyRefSolid = PhaseTPFunctorBuilder('S_excess',
                                             Excess_Solid_Entropy_Ref_Solid.functor,
                                             Excess_Liquid_Entropy_Ref_Solid.functor,
                                             Excess_Gas_Entropy_Ref_Solid.functor)

ExcessEntropyRefGas = PhaseTPFunctorBuilder('S_excess',
                                           Excess_Solid_Entropy_Ref_Gas.functor,
                                           Excess_Liquid_Entropy_Ref_Gas.functor,
                                           Excess_Gas_Entropy_Ref_Gas.functor)