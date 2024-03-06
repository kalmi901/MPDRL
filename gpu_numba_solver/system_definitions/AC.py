import numba as nb
from numba import cuda
import math


__AC_FUN_SIG = nb.float64(nb.float64, nb.float64, nb.float64[:], nb.float64[:])
def setup(ac_field, k):
    global _PA, _PAT, _GRADP, _UAC

    if ac_field == "CONST":
        """
        Homogeneous Pressure Field
        """

        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _PA(t, x, sp, dp):
            p = 0.0
            for i in range(k):
                p += dp[i] * math.sin(2*math.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])

            return p
        

        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _PAT(t, x, sp, dp):
            pt = 0.0
            for i in range(k):
                pt+= dp[i] * dp[k + i] \
                            * math.cos(2*math.pi*sp[1]*dp[  k + i] * t + dp[2*k + i])
                
            return pt
        
        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _GRADP(t, x, sp, dp):
            return 0.0
        
        @cuda.jit(__AC_FUN_SIG, device=True, inline=True)
        def _UAC(t, x, sp, dp):
            return 0.0
        
    elif ac_field == "SW_N":
        pass

    elif ac_field == "SN_A":
        pass

    else:
        print("Define a valid acoustic field type!")