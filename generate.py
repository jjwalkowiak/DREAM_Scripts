#!/usr/bin/env python3

import numpy as np

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.Solver as Solver
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.RunawayElectrons as RunawayElectrons
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.OhmicCurrent as OhmicCurrent
from DREAM.Formulas import getED, getEc
import DREAM.runiface


def kinetic_simulation_setup(T=1e3, nD=5e19, Cz=0.1, E_rel=0.1, Z0=1,
                             Nxi=30, Np=100, Pmax=None, printPmax=False, FLUX_LIMITER=True):
    """
    Generate settings for kinetic simulation to evaluate Dreicer electric field in plasma with tungsten impurities
    :param T: Plasma temperature (eV)
    :param nD: Deuterium density (m^-3)
    :param Cz: Impurity concentration - Impurity density = Cz * nD
    :param E_rel: Electric field as a fraction of Dreicer electric field
    :param Z0: Ionization state of impurity
    :param Nxi: Momentum grid resolution for pitch angle
    :param Np: Momentum grid resolution for momentum
    :param Pmax: Max value from momentum grid - by default calculated from
    :param printPmax: max(15*p_th, 2*p_c)
    :param FLUX_LIMITER: switch AdvectionInterpolationMethod to Flux limiter and solver to nonlinear
    :return: ds
    """
    # Create DREAM setting object
    ds = DREAMSettings()
    # Select collisions model
    ds.collisions.collfreq_mode       = Collisions.COLLFREQ_MODE_FULL
    ds.collisions.collfreq_type       = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds.collisions.bremsstrahlung_mode = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER
    ds.collisions.lnlambda            = Collisions.LNLAMBDA_ENERGY_DEPENDENT
    ds.collisions.pstar_mode          = Collisions.PSTAR_MODE_COLLISIONAL #_COLLISIONLESS

    # Set up radial grid
    ds.radialgrid.setMinorRadius(1)  # Plasma minor radius
    ds.radialgrid.setWallRadius(1)  # Tokamak wall minor radius
    ds.radialgrid.setNr(1)  # Number of grid cells
    ds.radialgrid.setB0(1)  # Magnetic field strength

    # Plasma setup
    nz = Cz * 0.1   # Impurity density

    # Set temperature
    ds.eqsys.T_cold.setPrescribedData(T)

    # Set ions - D background plasma
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=nD)
    # Set ions - W impurities
    ds.eqsys.n_i.addIon(name='W', Z=74, iontype=Ions.IONS_PRESCRIBED, Z0=Z0, n=nz)
    # Get electron density
    nfree = ds.eqsys.n_i.getFreeElectronDensity()[0][0]

    # Set E_field
    # E = 0.6  # Electric field strength (V/m)
    E = E_rel * getED(T, nfree)
    ds.eqsys.E_field.setPrescribedData(E)

    # Set momentum grid
    ds.runawaygrid.setEnabled(False)
    ds.hottailgrid.setEnabled(True)
    ds.hottailgrid.setNxi(Nxi)
    ds.hottailgrid.setNp(Np)
    # For pMax, you will want to make sure that it's somewhat above the critical momentum p_c ~ 1/sqrt(E/Ec-1).
    # Also, it should be at least ~7*p_th, where p_th=sqrt(2*T/511e3) is the thermal momentum,
    # to ensure that all of the original Maxwellian is within the momentum grid.
    # I would imagine that a condition like "max(10*p_th, 2*p_c)" could work fairly well.
    if Pmax is None:
        Pmax = max(15 * np.sqrt(T/5.11e5), 2*(E/getEc(T, nfree))**-0.5)
        if printPmax: print('\nPmax = '+str(Pmax) + '\nPth = '+str(np.sqrt(T/5.11e5)) + '\nPth = '+str((E/getEc(T, nfree))**-0.5))
    ds.hottailgrid.setPmax(Pmax)

    ds.eqsys.f_hot.setInitialProfiles(n0=nfree, T0=T)

    # OPTIONAL - set corrected conductivity
    ds.eqsys.j_ohm.setCorrectedConductivity(True)

    # Fluid Dreicer must be disabled whenever a distribution function is evolved
    ds.eqsys.n_re.setDreicer(RunawayElectrons.DREICER_RATE_DISABLED)


    #Set solver
    if FLUX_LIMITER:
        # Use Flux limiter
        ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=DistFunc.AD_INTERP_TCDF)
        ds.solver.setType(Solver.NONLINEAR)
    else:
        # Use the linear solver
        ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=DistFunc.AD_INTERP_UPWIND_2ND_ORDER)
        # ds.eqsys.f_hot.setAdvectionInterpolationMethod(ad_int=DistFunc.AD_INTERP_UPWIND)
        ds.solver.setType(Solver.LINEAR_IMPLICIT)

    # Set time stepper
    ds.timestep.setTmax(1.0e-6)
    ds.timestep.setNt(1)

    # Include information about time spent in different parts of the code...
    ds.output.setTiming(True, True)

    ds.other.include('fluid', 'scalar')

    # Save settings to HDF5 file
    ds.save('dream_settings.h5')
    return ds


def fluid_simulation(T=1e3, nD=5e19, Cz=0.1, E_rel=0.1, Z0=1, RE_Dreicer_NN=False):
    """
    Generate settings for kinetic simulation to evaluate Dreicer electric field in plasma with tungsten impurities
    :param T: Plasma temperature (eV)
    :param nD: Deuterium density (m^-3)
    :param Cz: Impurity concentration - Impurity density = Cz * nD
    :param E_rel: Electric field as a fraction of Dreicer electric field
    :param Z0: Ionization state of impurity
    :param RE_Dreicer_NN: When set to True, use neural network to evaluate Dreicer generation
    :return: ds
    """
    # Create DREAM setting object
    ds = DREAMSettings()
    # Select collisions model
    ds.collisions.collfreq_mode       = Collisions.COLLFREQ_MODE_FULL
    ds.collisions.collfreq_type       = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds.collisions.bremsstrahlung_mode = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER
    ds.collisions.lnlambda            = Collisions.LNLAMBDA_ENERGY_DEPENDENT
    ds.collisions.pstar_mode          = Collisions.PSTAR_MODE_COLLISIONAL #_COLLISIONLESS

    # Set runaway sources
    if RE_Dreicer_NN:
        ds.eqsys.n_re.setDreicer(RunawayElectrons.DREICER_RATE_NEURAL_NETWORK)
    else:
        ds.eqsys.n_re.setDreicer(RunawayElectrons.DREICER_RATE_CONNOR_HASTIE)

    # Set up radial grid
    ds.radialgrid.setMinorRadius(1)  # Plasma minor radius
    ds.radialgrid.setWallRadius(1)  # Tokamak wall minor radius
    ds.radialgrid.setNr(1)  # Number of grid cells
    ds.radialgrid.setB0(1)  # Magnetic field strength

    # Plasma setup
    nz = Cz * 0.1   # Impurity density

    # Set temperature
    ds.eqsys.T_cold.setPrescribedData(T)

    # Set ions - D background plasma
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=nD)
    # Set ions - W impurities
    ds.eqsys.n_i.addIon(name='W', Z=74, iontype=Ions.IONS_PRESCRIBED, Z0=Z0, n=nz)
    # Get electron density
    nfree = ds.eqsys.n_i.getFreeElectronDensity()[0][0]

    # Set E_field
    # E = 0.6  # Electric field strength (V/m)
    E = E_rel * getED(T, nfree)
    ds.eqsys.E_field.setPrescribedData(E)

    # Disable momentum grid
    ds.runawaygrid.setEnabled(False)
    ds.hottailgrid.setEnabled(False)

    # Set conductivity
    ds.eqsys.j_ohm.setConductivityMode(OhmicCurrent.CONDUCTIVITY_MODE_SAUTER_COLLISIONAL)

    # Set solver
    ds.solver.setType(Solver.NONLINEAR)

    # Set time stepper
    ds.timestep.setTmax(1.0e-6)
    ds.timestep.setNt(1)

    # Include information about time spent in different parts of the code...
    ds.output.setTiming(True, True)

    ds.other.include('fluid', 'scalar')

    # Save settings to HDF5 file
    ds.save('dream_settings.h5')
    return ds


def convergence_scan(arg_name, arg_value_list):
    """
    Run simulation with differnt settings
    :param arg_name: name of the varied parameter
    :param arg_value_list: list of varied parameter values used as input in simulations
    :return:
    """
    for arg_value in arg_value_list:
        simu_input = { arg_name: arg_value }
        ds = kinetic_simulation_setup(**simu_input)
        outputfile = 'output.h5'
        # Run simulation
        do = DREAM.runiface(ds, outputfile, timeout=600, quiet=True)
        # Read Dreicer generation
        gamma_Dreicer = do.other.fluid.gammaDreicer
        RE_rate = do.other.fluid.runawayRate
        # print output
        print('\n'+str(simu_input)+'\ngammaDreicer:\t' + str(*gamma_Dreicer))
        print('Total RE rate:\t' + str(*RE_rate))
        # print('\n'+str(simu_input)+'\ngammaDreicer:\t' + str([min(*gamma_Dreicer), max(*gamma_Dreicer)]))
        # Close output
        do.close()


def run_fluid(NN=False):
    """
    Run fluid simulation
    :param NN: When set to True, use neural network to evaluate Dreicer generation
    :return:
    """
    ds = fluid_simulation(RE_Dreicer_NN=NN)
    outputfile = 'output.h5'
    # Run simulation
    do = DREAM.runiface(ds, outputfile, timeout=600, quiet=True)
    # Read Dreicer generation
    gamma_Dreicer = do.other.fluid.gammaDreicer
    RE_rate = do.other.fluid.runawayRate
    # print output
    print('gammaDreicer:\t' + str(*gamma_Dreicer))
    print('Total RE rate:\t' + str(*RE_rate))
    # print('\n'+str(simu_input)+'\ngammaDreicer:\t' + str([min(*gamma_Dreicer), max(*gamma_Dreicer)]))
    # Close output
    do.close()


Np_list = [10, 30, 100, 300, 1000]
convergence_scan('Np', Np_list)
Nxi_list = [10, 30, 100]
convergence_scan('Nxi', Nxi_list)
convergence_scan('FLUX_LIMITER', [True, False])

print('\nfluid Connor-Hastie')
run_fluid()
print('\nfluid neural network')
run_fluid(NN=True)
