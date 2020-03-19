
    def max_t_a_root_function(self, t_a):
        v = self.fish_water_velocity
        tau_a = self.tau_times_thrust / self.thrust_a
        st = s_t(self.initial_speed, self.thrust_a, t_a, tau_a)
        value = -v * (self.initial_t + t_a) - (self.initial_x - st)
        return value
        
    def compute_max_t_a(self): # using bisection method, though Newton's should work for this function too
        a = 0
        b = 1000
        f_of_a = self.max_t_a_root_function(a)
        f_of_b = self.max_t_a_root_function(b)
        if np.sign(f_of_a) == np.sign(f_of_b):
            print("WARNING: Root function for max_t_a cannot converge.")
            print("f_of_a is ")
            print(f_of_a)
            print("f_of_b is ")
            print(f_of_b)
            if self.initial_speed > self.fish_water_velocity:
                print("Initial speed is faster than focal velocity.")
            else:
                print("Initial speed is slower than focal velocity.")
            return np.nan
        max_t_a  = np.nan
        tolerance = 1e-4
        max_iterations = 100
        for i in range(max_iterations):
            c = (a + b) / 2.0
            f_of_a = self.max_t_a_root_function(a)
            f_of_c = self.max_t_a_root_function(c)
            if abs(f_of_c) < tolerance:
                max_t_a = c
                print("Value of max_t_a converged within tolerance! It is:")
                print(max_t_a)
                break
            if np.sign(f_of_c) == np.sign(f_of_a):
                a = c
            else:
                b = c
        if not np.isfinite(max_t_a):
            print("WARNING: Root function for max_t_a should have been able to converge, but didn't.")
        return max_t_a
                
    def thrust_b_root_function(self, thrust_b): # Function to numerically minimize to make the fish end up at its focal point
        v = self.fish_water_velocity
        tau_b = self.tau_times_thrust / thrust_b
        tu = t_u(self.final_speed_a, thrust_b, v, tau_b)
        st = s_t(self.final_speed_a, thrust_b, tu, tau_b)
        ending_fish_position_in_water_coords = self.initial_x - self.length_a - st
        ending_focal_position_in_water_coords = -v * (self.initial_t + self.duration_a + tu)
        return ending_fish_position_in_water_coords - ending_focal_position_in_water_coords
        
    def compute_thrust_b(self):
        '''
        NOTE: THERE MAY BE A LARGE-T APPROXIMATION TO ELIMININATE THE NEED FOR THIS ITERATION IN MOST CASES
        To-do for the morning includes:
        
        - Large-t approximation to eliminate iterative root-finding for thrust under some circumstances.
        - Figure out why sometimes I got t_a_min > t_a_max above, and if it matches up to the condition for that I derived in Mathematica, which 
          seems to work in the opposite directon from what I need/expect. 
        - Figuring out whether t_a_max needs to account for the time it takes to slow down at thrust 0 in 
        this next step, 
        - Figuring out when to just ignore step A altogether
        - Figuring out what to do about step a when the thrust coming into it and therefore going out is all really high. 
        - Also, see if I can come up with conditions under which the large-t approximations are always good enough... like maybe the fish
        doesn't precisely match its focal point/velocity in the final step but it comes so close every time that
        the difference doesn't really matter energetically or conceptually. Basically need to explore the sensitivity
        to the difference between the exact iterative method and the approximate methods. The difference should be 
        pretty stark in cases of very low t where the approximation gives things like negative distances, but 
        perhaps it's possible to just cut off those cases altogether by assuming different behavior, or a limit that 
        is a bit more conservative than needed but avoids that, or something else.
        
        '''
        print("Starting to compute thrust b")
        v = self.fish_water_velocity
        eps = 1e-8
        if self.initial_speed > v: # fish needs to slow down, pick an initial guess between 0 and v
            min_thrust_b = eps
            max_thrust_b = v - eps
        else: # fish needs to speed up, pick thrust between v and max_thrust
            min_thrust_b = v + eps # add a bit to enforce thrust > v not >= v
            max_thrust_b = self.fish_max_thrust
        # Finding the root using the bisection method https://en.wikipedia.org/wiki/Bisection_method
        a = min_thrust_b
        b = max_thrust_b
        print("computing f_of_a and f_of_b")
        f_of_a = self.thrust_b_root_function(a)
        f_of_b = self.thrust_b_root_function(b)
        print(f_of_a)
        print(f_of_b)
        if np.sign(f_of_a) == np.sign(f_of_b):
            print("FINAL SEGMENT THRUST CANNOT CONVERGE.")
            self.print_root_function_inputs()
            return np.nan        
        thrust_b = np.nan
        tolerance = 1e-4
        max_iterations = 100
        for i in range(max_iterations):
            print("doing a step of compute_thrust_b, printing f_of_c")
            c = (a + b) / 2.0
            f_of_a = self.thrust_b_root_function(a)
            f_of_c = self.thrust_b_root_function(c)
            print(f_of_c)
            if abs(f_of_c) < tolerance:
                thrust_b = c
                print("Converged within tolerance!!!")
                #self.print_root_function_inputs()
                break
            if np.sign(f_of_c) == np.sign(f_of_a):
                a = c
            else:
                b = c
        if not np.isfinite(thrust_b):
            print("SHOULD NOT BE SEEING THIS -- Convergence somehow failed even though the condition to try was met. f_of_c was:")
            print(f_of_c)
            self.print_root_function_inputs()
            return np.nan  
        return thrust_b      
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
    #def thrust_root_function(self, u_f): # Function to numerically minimize to make the fish end up at its focal point
    #    v = self.fish_water_velocity
    #    tu = t_u(self.u_i, u_f, v, self.calculate_tau(u_f))
    #    st = s_t(self.u_i, u_f, tu, self.calculate_tau(u_f))
    #    ending_fish_position_in_water_coords = self.initial_x - st
    #    ending_focal_position_in_water_coords = -v * (self.initial_t + tu)
    #    return ending_fish_position_in_water_coords - ending_focal_position_in_water_coords
                
    def print_root_function_inputs(self):
        # for relatively easy copying to Mathematica to test
        print("--------start of root function values--------")
        print("final_speed_a")
        print(self.final_speed_a)
        print("v")
        print(self.fish_water_velocity)
        print("tau_times_thrust")
        print(self.tau_times_thrust) # using dummy u_f variable here because cancel it out
        print("xp")
        print(self.initial_x)
        print("tp")
        print(self.initial_t)
        print("--------end of root function values--------")
        
        
        
            #def compute_matching_thrust(self):
    #    # Uses the Newton-Raphson method to find the root of thrust_root_function, which corresponds to finding the single thrust value that allows the fish starting at initial_x
    #    # and initial_t to rejoin its focal point while moving at its focal velocity, at which point it can switch thrust to its focal velocity to hold position.
    #    v = self.fish_water_velocity
    #    eps = 1e-8
    #    if self.u_i > v: # fish needs to slow down, pick an initial guess between 0 and v
    #        min_thrust = eps
    #        max_thrust = v - eps
    #    else: # fish needs to speed up, pick thrust between v and max_thrust
    #        min_thrust = v + eps # add a bit to enforce thrust > v not >= v
    #        max_thrust = self.fish_max_thrust
    #    # Finding the root using the bisection method
    #    # https://en.wikipedia.org/wiki/Bisection_method
    #    a = min_thrust
    #    b = max_thrust
    #    f_of_a = self.thrust_root_function(a)
    #    f_of_b = self.thrust_root_function(b)
    #    if np.sign(f_of_a) == np.sign(f_of_b):
    #        print("FINAL SEGMENT THRUST CANNOT CONVERGE.")
    #        self.print_root_function_inputs()
    #        return np.nan        
    #    
    #    thrust = np.nan
    #    step = 1
    #    tolerance = 1e-4
    #    max_iterations = 100
    #    while step < max_iterations:
    #        print("doing a step, printing f_of_c")
    #        c = (a + b) / 2
    #        f_of_a = self.thrust_root_function(a)
    #        f_of_c = self.thrust_root_function(c)
    #        print(f_of_c)
    #        if abs(f_of_c) < tolerance:
    #            thrust = c
    #            print("Converged within tolerance!!!")
    #            self.print_root_function_inputs()
    #            break
    #        step += 1
    #        if np.sign(f_of_c) == np.sign(f_of_a):
    #            a = c
    #        else:
    #            b = c
    #    if not np.isfinite(thrust):
    #        print("SHOULD NOT BE SEEING THIS -- Convergence somehow failed even though the condition to try was met. f_of_c was:")
    #        print(f_of_c)
    #        self.print_root_function_inputs()
    #        return np.nan
    #    else:
    #        #thrust = (min_thrust + max_thrust) / 2
    #        #for i in range(newton_iterations):
    #        #    while thrust < min_thrust or thrust > max_thrust:
    #        #        if thrust < min_thrust:
    #        #            thrust = min_thrust + (min_thrust - thrust)
    #        #        if thrust > max_thrust:
    #        #            thrust = max_thrust - (thrust - max_thrust)
    #        #    print(thrust)
    #        #    thrust = thrust - self.thrust_root_function(thrust) / self.thrust_root_function_derivative(thrust)
    #        print("returning final thrust")
    #        print(thrust)
    #        print("its root function value is")
    #        print(self.thrust_root_function(thrust))
    #        return thrust    
    
        
        
# Tests with convergence tolerances

# Changing them to 1e-4
# Out of 30x 3000-iteration tests completed in 0.58432 s/run, fitnesses were -0.56981 +/- 0.05899 (range -0.79743 to -0.52937) in durations 7.21038 +/- 0.62443 sd (range 6.25433 to 9.81079)

# Now 1e-5
# Out of 100x 3000-iteration tests completed in 0.61919 s/run, fitnesses were -0.55787 +/- 0.03259 (range -0.79212 to -0.52083) in durations 7.05837 +/- 0.41209 sd (range 6.23500 to 8.35322)

# Noew 1e-6
# Out of 30x 3000-iteration tests completed in 0.58842 s/run, fitnesses were -0.56091 +/- 0.04091 (range -0.70400 to -0.53046) in durations 6.99802 +/- 0.42118 sd (range 6.11190 to 8.09802)
# Out of 30x 3000-iteration tests completed in 0.59422 s/run, fitnesses were -0.57145 +/- 0.05685 (range -0.83421 to -0.53740) in durations 7.04572 +/- 0.47572 sd (range 6.31895 to 8.06357)

# With convergence tolerances 1e-7
# Out of 30x 3000-iteration tests completed in 0.63653 s/run, fitnesses were -0.55125 +/- 0.01324 (range -0.58724 to -0.52704) in durations 6.90202 +/- 0.47795 sd (range 6.19926 to 7.72941)
# Out of 30x 3000-iteration tests completed in 0.63928 s/run, fitnesses were -0.55618 +/- 0.02875 (range -0.65943 to -0.52737) in durations 6.93965 +/- 0.54151 sd (range 6.00348 to 8.64727)
# Out of 100x 3000-iteration tests completed in 0.60840 s/run, fitnesses were -0.55953 +/- 0.04297 (range -0.79936 to -0.52280) in durations 7.08297 +/- 0.52215 sd (range 6.15781 to 9.26930)

# Now 1e-8 with max_newton_iterations upped to 100
# Out of 100x 3000-iteration tests completed in 0.64707 s/run, fitnesses were -0.56097 +/- 0.03966 (range -0.73374 to -0.52434) in durations 7.08979 +/- 0.45873 sd (range 6.01597 to 8.54822)

# Now 1e-9
# Out of 100x 3000-iteration tests completed in 0.68225 s/run, fitnesses were -0.56176 +/- 0.04788 (range -0.89078 to -0.52319) in durations 7.02223 +/- 0.54926 sd (range 6.17467 to 9.41248)

# Conclusion on convergence tolerances: It doesn't seem they matter all that much. It seems 1e-7 provides good timing. COMPARE BELOW WITH THOSE VALUES

# Now changing mixing ratio:

# Now at 0.5
# Out of 100x 3000-iteration tests completed in 0.63584 s/run, fitnesses were -0.55461 +/- 0.02286 (range -0.63058 to -0.52111) in durations 7.02493 +/- 0.47629 sd (range 6.05049 to 9.43144)

# Now at 0.7
# # Out of 100x 3000-iteration tests completed in 0.60840 s/run, fitnesses were -0.55953 +/- 0.04297 (range -0.79936 to -0.52280) in durations 7.08297 +/- 0.52215 sd (range 6.15781 to 9.26930)

# Now at 0.9
# Out of 100x 3000-iteration tests completed in 0.62739 s/run, fitnesses were -0.56515 +/- 0.05812 (range -1.02563 to -0.52521) in durations 7.16899 +/- 0.64459 sd (range 6.17557 to 10.10652)

# Now at 0.99
# Out of 100x 3000-iteration tests completed in 0.59853 s/run, fitnesses were -0.56152 +/- 0.04332 (range -0.83011 to -0.52684) in durations 7.04263 +/- 0.45644 sd (range 6.22032 to 8.90339)

# Conclusion on mixing ratio: It doesn't seem to matter all that much, with both 0.5 and 0.99 providing comparable results.

# Changing mutation scale

# At 0.01 -- clearly terrible!
# Out of 100x 3000-iteration tests completed in 0.52919 s/run, fitnesses were -0.88309 +/- 0.12665 (range -1.22246 to -0.61933) in durations 7.73425 +/- 0.84139 sd (range 4.98677 to 9.60531)

# At 0.05
# Out of 1000x 3000-iteration tests completed in 0.62093 s/run, fitnesses were -0.60621 +/- 0.09452 (range -1.03668 to -0.52268) in durations 7.13523 +/- 0.56237 sd (range 5.72035 to 9.93134)

# At 0.1
# Out of 100x  3000-iteration tests completed in 0.63584 s/run, fitnesses were -0.55461 +/- 0.02286 (range -0.63058 to -0.52111) in durations 7.02493 +/- 0.47629 sd (range 6.05049 to 9.43144)
# Out of 1000x 3000-iteration tests completed in 0.70909 s/run, fitnesses were -0.56056 +/- 0.04173 (range -0.95752 to -0.51935) in durations 7.08527 +/- 0.52068 sd (range 6.02738 to 9.71533)

# At 0.15
# Out of 1000x 3000-iteration tests completed in 0.74810 s/run, fitnesses were -0.55882 +/- 0.03501 (range -0.79471 to -0.52080) in durations 7.10089 +/- 0.51301 sd (range 5.76788 to 9.92390)

# 0.2
# Out of 100x 3000-iteration tests completed in 0.64212 s/run, fitnesses were -0.56008 +/- 0.03335 (range -0.77899 to -0.52542) in durations 7.07225 +/- 0.47134 sd (range 6.24463 to 8.41062)

# At 0.3
# Out of 100x 3000-iteration tests completed in 0.65174 s/run, fitnesses were -0.56336 +/- 0.03062 (range -0.74936 to -0.52029) in durations 7.17658 +/- 0.51393 sd (range 6.13471 to 8.67253)

# Conclusion: Clearly 0.05 is worse and 0.15 a little bit better than 0.10, but not by much. 

# Now trying mutation rate

# Now a rate of 0.0 -- tried by accident thanks to a typo -- awful!!!
# Out of 1000x 3000-iteration tests completed in 0.52076 s/run, fitnesses were -1.12677 +/- 0.13688 (range -1.58018 to -0.69431) in durations 8.18469 +/- 1.18743 sd (range 3.82575 to 12.12693)

# Now a rate of 0.05 -- wow, much worse!
# Out of 1000x 3000-iteration tests completed in 0.62933 s/run, fitnesses were -0.67223 +/- 0.13785 (range -1.20988 to -0.52283) in durations 7.34338 +/- 0.69831 sd (range 4.85360 to 10.29808)

# With a rate of 0.15
# # Out of 1000x 3000-iteration tests completed in 0.74810 s/run, fitnesses were -0.55882 +/- 0.03501 (range -0.79471 to -0.52080) in durations 7.10089 +/- 0.51301 sd (range 5.76788 to 9.92390)

# Now 0.25
# Out of 1000x 3000-iteration tests completed in 0.69671 s/run, fitnesses were -0.54388 +/- 0.01732 (range -0.67701 to -0.52047) in durations 7.16707 +/- 0.42847 sd (range 5.67682 to 9.55835)

# Now 0.3
# Out of 1000x 3000-iteration tests completed in 0.61678 s/run, fitnesses were -0.54290 +/- 0.01851 (range -0.73295 to -0.51959) in durations 7.18462 +/- 0.40520 sd (range 6.12070 to 9.22067)

# Now 0.35 -- the sweet spot
# Out of 1000x 3000-iteration tests completed in 0.67307 s/run, fitnesses were -0.54128 +/- 0.01511 (range -0.68215 to -0.52082) in durations 7.23763 +/- 0.41964 sd (range 5.98753 to 9.53983)

# Now 0.4
# Out of 1000x 3000-iteration tests completed in 0.59136 s/run, fitnesses were -0.54318 +/- 0.01931 (range -0.69291 to -0.52024) in durations 7.24963 +/- 0.43107 sd (range 6.08416 to 9.94160)

# Now 0.5
# Out of 1000x 3000-iteration tests completed in 0.54258 s/run, fitnesses were -0.54859 +/- 0.02832 (range -0.79768 to -0.52238) in durations 7.27751 +/- 0.46434 sd (range 5.63789 to 10.14806)

# Now 0.7 
# Out of 1000x 3000-iteration tests completed in 0.51071 s/run, fitnesses were -0.56503 +/- 0.03652 (range -0.75722 to -0.53099) in durations 7.34196 +/- 0.58248 sd (range 5.56701 to 10.86672)

# Conclusion -- Mutation rate of 0.35 seems to be ideal.

# Now investigating the shuffle interval

# Now 10
# Out of 1000x 3000-iteration tests completed in 0.59369 s/run, fitnesses were -0.53999 +/- 0.01632 (range -0.69959 to -0.52081) in durations 7.22919 +/- 0.37054 sd (range 5.99994 to 9.51462)

# Now 15 -- sticking with 15, seems to be in the middle of a good range if not the absolute best on this one test problem
# Out of 1000x 3000-iteration tests completed in 0.67365 s/run, fitnesses were -0.54239 +/- 0.02086 (range -0.72651 to -0.52201) in durations 7.23602 +/- 0.40225 sd (range 5.64913 to 9.64219)

# Now 20 (from above)
# Out of 1000x 3000-iteration tests completed in 0.67307 s/run, fitnesses were -0.54128 +/- 0.01511 (range -0.68215 to -0.52082) in durations 7.23763 +/- 0.41964 sd (range 5.98753 to 9.53983)

# Now 100
# Out of 1000x 3000-iteration tests completed in 0.66693 s/run, fitnesses were -0.54394 +/- 0.02124 (range -0.85579 to -0.52213) in durations 7.23215 +/- 0.42798 sd (range 5.83846 to 9.63836)

# After that, popsize!! Trying to stick with iteration/population combinations that should give roughly the same computaton time

# 3000-iterations with popsize = 4 (from above)
# Out of 1000x 3000-iteration tests completed in 0.67365 s/run, fitnesses were -0.54239 +/- 0.02086 (range -0.72651 to -0.52201) in durations 7.23602 +/- 0.40225 sd (range 5.64913 to 9.64219)

# 2000-iterations with popsize = 6
# Out of 1000x 2000-iteration tests completed in 0.65404 s/run, fitnesses were -0.54804 +/- 0.02510 (range -0.79877 to -0.52189) in durations 7.23080 +/- 0.46523 sd (range 5.93550 to 9.91183)

# 1500-iterations with popsize = 8
# Out of 1000x 1500-iteration tests completed in 0.61451 s/run, fitnesses were -0.55523 +/- 0.02869 (range -0.73189 to -0.52422) in durations 7.19216 +/- 0.48957 sd (range 5.61987 to 9.32904)

# Conclusion on this: Sticking with popsize 4 and greater number of iterations.


# Now trying Cuckoo with double the N (100), half the iterations (500) -- Not shabby, but not much improved
#Lowest energy cost after 500 CS iterations (  200202 evaluations) was   0.159188 joules. Mean speed 29.2 cm/s,  1.09 bodylengths/s. Metabolic rate    74.5 mg O2/kg/hr ( 2.3X SMR). 
#Lowest energy cost after 500 CS iterations (  200202 evaluations) was   0.161499 joules. Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    74.2 mg O2/kg/hr ( 2.3X SMR). 
#Lowest energy cost after 500 CS iterations (  200202 evaluations) was   0.162621 joules. Mean speed 28.7 cm/s,  1.07 bodylengths/s. Metabolic rate    71.4 mg O2/kg/hr ( 2.3X SMR). 
#Lowest energy cost after 500 CS iterations (  200202 evaluations) was   0.160536 joules. Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    72.6 mg O2/kg/hr ( 2.3X SMR). 
#Lowest energy cost after 500 CS iterations (  200202 evaluations) was   0.162495 joules. Mean speed 28.6 cm/s,  1.07 bodylengths/s. Metabolic rate    70.7 mg O2/kg/hr ( 2.2X SMR). 

# Now trying Cuckoo with half the N (25), twice the iterations (2000)
#Lowest energy cost after 2000 CS iterations (  200052 evaluations) was   0.155232 joules. Mean speed 29.4 cm/s,  1.10 bodylengths/s. Metabolic rate    76.0 mg O2/kg/hr ( 2.3X SMR). 
#Lowest energy cost after 2000 CS iterations (  200052 evaluations) was   0.168859 joules. Mean speed 28.2 cm/s,  1.05 bodylengths/s. Metabolic rate    66.8 mg O2/kg/hr ( 2.2X SMR). 
#Lowest energy cost after 2000 CS iterations (  200052 evaluations) was   0.161044 joules. Mean speed 28.8 cm/s,  1.08 bodylengths/s. Metabolic rate    72.4 mg O2/kg/hr ( 2.3X SMR). 
#Lowest energy cost after 2000 CS iterations (  200052 evaluations) was   0.172687 joules. Mean speed 26.7 cm/s,  1.00 bodylengths/s. Metabolic rate    42.8 mg O2/kg/hr ( 1.8X SMR). 
#Lowest energy cost after 2000 CS iterations (  200052 evaluations) was   0.159589 joules. Mean speed 28.7 cm/s,  1.07 bodylengths/s. Metabolic rate    69.1 mg O2/kg/hr ( 2.2X SMR). 

# Looks like the Salp algorithm is not the worst of the bunch, but not impressing compared to Cuckoo or microGA... maybe third best? Worth consideration.
#Lowest energy cost after 2000 CS iterations (  200002 evaluations) was   0.176184 joules. Mean speed 29.2 cm/s,  1.09 bodylengths/s. Metabolic rate    82.8 mg O2/kg/hr ( 2.5X SMR). 
#Lowest energy cost after 2000 CS iterations (  200002 evaluations) was   0.205302 joules. Mean speed 27.7 cm/s,  1.03 bodylengths/s. Metabolic rate    70.3 mg O2/kg/hr ( 2.2X SMR). 
#Lowest energy cost after 2000 CS iterations (  200002 evaluations) was   0.161861 joules. Mean speed 29.3 cm/s,  1.09 bodylengths/s. Metabolic rate    77.6 mg O2/kg/hr ( 2.4X SMR). 
#Lowest energy cost after 2000 CS iterations (  200002 evaluations) was   0.164316 joules. Mean speed 28.7 cm/s,  1.07 bodylengths/s. Metabolic rate    72.9 mg O2/kg/hr ( 2.3X SMR). 
#Lowest energy cost after 2000 CS iterations (  200002 evaluations) was   0.184262 joules. Mean speed 26.7 cm/s,  1.00 bodylengths/s. Metabolic rate    46.8 mg O2/kg/hr ( 1.8X SMR). 

# Looks like the BAT algorithm is a no-go, or at least badly malfunctioning somehow (maybe my manually clipping the values invalidates them)
#Lowest energy cost after 2000 BAT iterations (  200102 evaluations) was  23.831078 joules. Mean speed 27.2 cm/s,  1.01 bodylengths/s. Metabolic rate   399.0 mg O2/kg/hr ( 8.0X SMR). 
#Lowest energy cost after 2000 BAT iterations (  200102 evaluations) was  15.934972 joules. Mean speed 25.3 cm/s,  0.95 bodylengths/s. Metabolic rate   483.4 mg O2/kg/hr ( 9.5X SMR). 
#Lowest energy cost after 2000 BAT iterations (  200102 evaluations) was 99999999999.455231 joules. Mean speed  1.0 cm/s,  0.04 bodylengths/s. Metabolic rate  1230.3 mg O2/kg/hr (22.6X SMR). 
#Lowest energy cost after 2000 BAT iterations (  200102 evaluations) was 128.416262 joules. Mean speed 26.2 cm/s,  0.98 bodylengths/s. Metabolic rate   639.8 mg O2/kg/hr (12.2X SMR). 
#Lowest energy cost after 2000 BAT iterations (  200102 evaluations) was 122.391682 joules. Mean speed 25.6 cm/s,  0.96 bodylengths/s. Metabolic rate   537.0 mg O2/kg/hr (10.4X SMR). 

# Whale optimization with 25 whales, 4000 iterations, is not too bad...
#Lowest energy cost after 4000 WOA iterations (  200002 evaluations) was   0.187764 joules. Mean speed 26.4 cm/s,  0.99 bodylengths/s. Metabolic rate    40.8 mg O2/kg/hr ( 1.7X SMR). 
#Lowest energy cost after 4000 WOA iterations (  200002 evaluations) was   0.201735 joules. Mean speed 26.2 cm/s,  0.98 bodylengths/s. Metabolic rate    39.2 mg O2/kg/hr ( 1.7X SMR). 
#Lowest energy cost after 4000 WOA iterations (  200002 evaluations) was   0.170264 joules. Mean speed 28.3 cm/s,  1.06 bodylengths/s. Metabolic rate    69.5 mg O2/kg/hr ( 2.2X SMR). 
#Lowest energy cost after 4000 WOA iterations (  200002 evaluations) was   0.200995 joules. Mean speed 26.2 cm/s,  0.98 bodylengths/s. Metabolic rate    39.7 mg O2/kg/hr ( 1.7X SMR). 
#Lowest energy cost after 4000 WOA iterations (  200002 evaluations) was   0.167459 joules. Mean speed 29.3 cm/s,  1.09 bodylengths/s. Metabolic rate    80.9 mg O2/kg/hr ( 2.4X SMR). 

# Dropping to 10 whales, 10000 iterations, doesn't work as well...
#Lowest energy cost after 10000 WOA iterations (  200002 evaluations) was   0.221468 joules. Mean speed 28.2 cm/s,  1.05 bodylengths/s. Metabolic rate    82.3 mg O2/kg/hr ( 2.4X SMR). 
#Lowest energy cost after 10000 WOA iterations (  200002 evaluations) was   0.214125 joules. Mean speed 26.4 cm/s,  0.98 bodylengths/s. Metabolic rate    45.9 mg O2/kg/hr ( 1.8X SMR). 
#Lowest energy cost after 10000 WOA iterations (  200002 evaluations) was   0.171314 joules. Mean speed 28.1 cm/s,  1.05 bodylengths/s. Metabolic rate    67.2 mg O2/kg/hr ( 2.2X SMR). 
#Lowest energy cost after 10000 WOA iterations (  200002 evaluations) was   0.230023 joules. Mean speed 26.2 cm/s,  0.98 bodylengths/s. Metabolic rate    45.5 mg O2/kg/hr ( 1.8X SMR). 
#Lowest energy cost after 10000 WOA iterations (  200002 evaluations) was   0.163248 joules. Mean speed 28.7 cm/s,  1.07 bodylengths/s. Metabolic rate    71.2 mg O2/kg/hr ( 2.3X SMR)

# PSO with 200 particles, 500 iterations -- some major malfunctions
#Lowest energy cost after 500 PSO iterations (  200002 evaluations) was 18356522638.799732 joules. Mean speed 59675675.4 cm/s, 2228949.89 bodylengths/s. Metabolic rate 6495283383760.0 mg O2/kg/hr (114096409547.4X SMR). 
#Lowest energy cost after 500 PSO iterations (  200002 evaluations) was   0.160829 joules. Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    73.3 mg O2/kg/hr ( 2.3X SMR). 
#Lowest energy cost after 500 PSO iterations (  200002 evaluations) was   0.160825 joules. Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    74.6 mg O2/kg/hr ( 2.3X SMR). 
#Lowest energy cost after 500 PSO iterations (  200002 evaluations) was 99999999999.152466 joules. Mean speed  1.0 cm/s,  0.04 bodylengths/s. Metabolic rate  1230.3 mg O2/kg/hr (22.6X SMR). 
#Lowest energy cost after 500 PSO iterations (  200002 evaluations) was   0.193212 joules. Mean speed 26.3 cm/s,  0.98 bodylengths/s. Metabolic rate    39.4 mg O2/kg/hr ( 1.7X SMR). 

# Particle swarm with 50 particles, 2000 iterations
# Lowest energy cost after 2000 PSO iterations (  200002 evaluations) was   0.246142 joules. Mean speed 26.9 cm/s,  1.01 bodylengths/s. Metabolic rate    71.2 mg O2/kg/hr ( 2.3X SMR). 
# Lowest energy cost after 2000 PSO iterations (  200002 evaluations) was   0.415273 joules. Mean speed 29.5 cm/s,  1.10 bodylengths/s. Metabolic rate   172.7 mg O2/kg/hr ( 4.0X SMR). 
# Lowest energy cost after 2000 PSO iterations (  200002 evaluations) was   0.160800 joules. Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    74.4 mg O2/kg/hr ( 2.3X SMR). 
# Lowest energy cost after 2000 PSO iterations (  200002 evaluations) was   0.247144 joules. Mean speed 27.4 cm/s,  1.02 bodylengths/s. Metabolic rate    78.9 mg O2/kg/hr ( 2.4X SMR). 
#Lowest energy cost after 2000 PSO iterations (  200002 evaluations) was   0.206700 joules. Mean speed 27.9 cm/s,  1.04 bodylengths/s. Metabolic rate    77.3 mg O2/kg/hr ( 2.4X SMR). 
#Lowest energy cost after 2000 PSO iterations (  200002 evaluations) was   0.182151 joules. Mean speed 26.8 cm/s,  1.00 bodylengths/s. Metabolic rate    48.0 mg O2/kg/hr ( 1.8X SMR). 
#Lowest energy cost after 2000 PSO iterations (  200002 evaluations) was   0.205036 joules. Mean speed 27.7 cm/s,  1.03 bodylengths/s. Metabolic rate    70.9 mg O2/kg/hr ( 2.2X SMR). 
#Lowest energy cost after 2000 PSO iterations (  200002 evaluations) was   0.152850 joules. Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    70.6 mg O2/kg/hr ( 2.2X SMR). 
#Lowest energy cost after 2000 PSO iterations (  200002 evaluations) was   0.418488 joules. Mean speed 28.8 cm/s,  1.08 bodylengths/s. Metabolic rate   155.7 mg O2/kg/hr ( 3.7X SMR). 

# Whale with 50 whales, 2000 iterations
# Lowest energy cost after 2000 WOA iterations (  200002 evaluations) was   0.195713 joules. Mean speed 26.6 cm/s,  0.99 bodylengths/s. Metabolic rate    46.3 mg O2/kg/hr ( 1.8X SMR). 
# Lowest energy cost after 2000 WOA iterations (  200002 evaluations) was   0.239173 joules. Mean speed 26.9 cm/s,  1.01 bodylengths/s. Metabolic rate    66.8 mg O2/kg/hr ( 2.2X SMR). 
# Lowest energy cost after 2000 WOA iterations (  200002 evaluations) was   0.188502 joules. Mean speed 26.8 cm/s,  1.00 bodylengths/s. Metabolic rate    48.0 mg O2/kg/hr ( 1.8X SMR). 
# Lowest energy cost after 2000 WOA iterations (  200002 evaluations) was   0.183356 joules. Mean speed 27.0 cm/s,  1.01 bodylengths/s. Metabolic rate    51.7 mg O2/kg/hr ( 1.9X SMR). 
# Lowest energy cost after 2000 WOA iterations (  200002 evaluations) was   0.257812 joules. Mean speed 29.3 cm/s,  1.10 bodylengths/s. Metabolic rate   124.6 mg O2/kg/hr ( 3.2X SMR). 

# Whale with 200 whales, 500 iterations
# Lowest energy cost after 500 WOA iterations (  200002 evaluations) was   0.218385 joules. Mean speed 27.7 cm/s,  1.04 bodylengths/s. Metabolic rate    78.7 mg O2/kg/hr ( 2.4X SMR). 
# Lowest energy cost after 500 WOA iterations (  200002 evaluations) was   0.167548 joules. Mean speed 29.3 cm/s,  1.10 bodylengths/s. Metabolic rate    81.1 mg O2/kg/hr ( 2.4X SMR). 
# Lowest energy cost after 500 WOA iterations (  200002 evaluations) was   0.192978 joules. Mean speed 26.3 cm/s,  0.98 bodylengths/s. Metabolic rate    38.4 mg O2/kg/hr ( 1.7X SMR). 
# Lowest energy cost after 500 WOA iterations (  200002 evaluations) was   0.185633 joules. Mean speed 26.6 cm/s,  0.99 bodylengths/s. Metabolic rate    43.3 mg O2/kg/hr ( 1.8X SMR). 
# Lowest energy cost after 500 WOA iterations (  200002 evaluations) was   0.217853 joules. Mean speed 28.3 cm/s,  1.06 bodylengths/s. Metabolic rate    88.5 mg O2/kg/hr ( 2.6X SMR). 

# Whale with 400 whales, 250 iterations
# Lowest energy cost after 250 WOA iterations (  200002 evaluations) was   0.175355 joules. Mean speed 29.8 cm/s,  1.11 bodylengths/s. Metabolic rate    88.7 mg O2/kg/hr ( 2.6X SMR). 
# Lowest energy cost after 250 WOA iterations (  200002 evaluations) was   0.199517 joules. Mean speed 27.0 cm/s,  1.01 bodylengths/s. Metabolic rate    57.2 mg O2/kg/hr ( 2.0X SMR). 
# Lowest energy cost after 250 WOA iterations (  200002 evaluations) was   0.177540 joules. Mean speed 29.2 cm/s,  1.09 bodylengths/s. Metabolic rate    83.9 mg O2/kg/hr ( 2.5X SMR). 
# Lowest energy cost after 250 WOA iterations (  200002 evaluations) was   0.216292 joules. Mean speed 26.7 cm/s,  1.00 bodylengths/s. Metabolic rate    54.5 mg O2/kg/hr ( 2.0X SMR). 
# Lowest energy cost after 250 WOA iterations (  200002 evaluations) was   0.184846 joules. Mean speed 26.7 cm/s,  1.00 bodylengths/s. Metabolic rate    47.0 mg O2/kg/hr ( 1.8X SMR). 

# Moth-flame is REALLY bad for this problem.
# Lowest energy cost after 2000 MFO iterations (  200002 evaluations) was   0.364881 joules. Mean speed 33.7 cm/s,  1.26 bodylengths/s. Metabolic rate   228.3 mg O2/kg/hr ( 5.0X SMR). 
# Lowest energy cost after 2000 MFO iterations (  200002 evaluations) was   0.236918 joules. Mean speed 27.3 cm/s,  1.02 bodylengths/s. Metabolic rate    74.7 mg O2/kg/hr ( 2.3X SMR). 
# Lowest energy cost after 2000 MFO iterations (  200002 evaluations) was   0.209976 joules. Mean speed 27.4 cm/s,  1.02 bodylengths/s. Metabolic rate    65.2 mg O2/kg/hr ( 2.1X SMR). 
# Lowest energy cost after 2000 MFO iterations (  200002 evaluations) was   0.255405 joules. Mean speed 26.7 cm/s,  1.00 bodylengths/s. Metabolic rate    64.8 mg O2/kg/hr ( 2.1X SMR). 
# Lowest energy cost after 2000 MFO iterations (  200002 evaluations) was   0.365912 joules. Mean speed 31.2 cm/s,  1.17 bodylengths/s. Metabolic rate   169.7 mg O2/kg/hr ( 4.0X SMR). 


# GWO is CLEARLY not working as well as my uGA algorithm. More iterations, worse convergence.
# Lowest energy cost after 20000 GWO iterations (  200002 evaluations) was   0.183044 joules. Mean speed 26.9 cm/s,  1.00 bodylengths/s. Metabolic rate    49.2 mg O2/kg/hr ( 1.9X SMR). 
# Lowest energy cost after 20000 GWO iterations was   0.162802 joules. Mean speed 28.8 cm/s,  1.08 bodylengths/s. Metabolic rate    73.1 mg O2/kg/hr ( 2.3X SMR). 
# Lowest energy cost after 20000 GWO iterations was   0.182616 joules. Mean speed 26.8 cm/s,  1.00 bodylengths/s. Metabolic rate    47.5 mg O2/kg/hr ( 1.8X SMR). 
# Lowest energy cost after 20000 GWO iterations was   0.161895 joules. Mean speed 28.8 cm/s,  1.07 bodylengths/s. Metabolic rate    71.8 mg O2/kg/hr ( 2.3X SMR). 
# Lowest energy cost after 20000 GWO iterations was   0.182865 joules. Mean speed 26.9 cm/s,  1.00 bodylengths/s. Metabolic rate    48.6 mg O2/kg/hr ( 1.9X SMR). 
# Lowest energy cost after 20000 GWO iterations was   0.162733 joules. Mean speed 28.5 cm/s,  1.07 bodylengths/s. Metabolic rate    69.5 mg O2/kg/hr ( 2.2X SMR). 
# Lowest energy cost after 20000 GWO iterations was   0.162242 joules. Mean speed 29.4 cm/s,  1.10 bodylengths/s. Metabolic rate    79.7 mg O2/kg/hr ( 2.4X SMR). s
# Lowest energy cost after 20000 GWO iterations was   0.171237 joules. Mean speed 26.7 cm/s,  1.00 bodylengths/s. Metabolic rate    43.8 mg O2/kg/hr ( 1.8X SMR). 
# Changing wolf count to 7 from default of 5... no substantial improvement.
# Lowest energy cost after 10000 GWO iterations (  140002 evaluations) was   0.183456 joules. Mean speed 26.8 cm/s,  1.00 bodylengths/s. Metabolic rate    47.0 mg O2/kg/hr ( 1.8X SMR). 

# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.154084 joules (origin: crossover). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    71.7 mg O2/kg/hr ( 2.3X SMR). 
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.155315 joules (origin: mutation ). Mean speed 29.4 cm/s,  1.10 bodylengths/s. Metabolic rate    75.1 mg O2/kg/hr ( 2.3X SMR). 
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.161781 joules (origin: crossover). Mean speed 28.9 cm/s,  1.08 bodylengths/s. Metabolic rate    73.0 mg O2/kg/hr ( 2.3X SMR). 
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.160911 joules (origin: crossover). Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    74.6 mg O2/kg/hr ( 2.3X SMR). 
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.155866 joules (origin: crossover). Mean speed 29.0 cm/s,  1.08 bodylengths/s. Metabolic rate    72.2 mg O2/kg/hr ( 2.3X SMR). 
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.162523 joules (origin: mutation ). Mean speed 29.1 cm/s,  1.09 bodylengths/s. Metabolic rate    75.1 mg O2/kg/hr ( 2.3X SMR).   
# Lowest energy cost after 15x1500+11529 uGA iterations (  200004 evaluations) was   0.164162 joules (origin: mutation ). Mean speed 28.2 cm/s,  1.05 bodylengths/s. Metabolic rate    65.7 mg O2/kg/hr ( 2.2X SMR).    
