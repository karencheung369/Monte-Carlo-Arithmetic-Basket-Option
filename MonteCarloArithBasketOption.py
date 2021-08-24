from scipy.stats import norm
import numpy as np

class MonteCarloArithBasketOption():
    """
    Monte Carlo method with control variate technique for arithmetic mean basket call/put options with two assets.
        s01 = asset 1 spot price
        s02 = asset 2 spot price
        K = strike price
        T = time to maturity (year)
        r = risk free interest rate
        sigma1 = volatility 1
        sigma2 = volatility 2
        rho = correlation
        optionType = call or put
        m = # of path in Monte Carlo simulation
        ctrlVar = with or without control variate
    """
    def __init__(self, s01=None, s02=None, r=0, T=0, K=None, sigma1=None, sigma2=None, rho=None, optionType=None, m=100000, ctrlVar=False):
        try:
          self.s01 = s01
          self.s02 = s02
          self.sigma1 = sigma1
          self.sigma2 = sigma2
          self.r = r
          self.T = T
          self.K = K
          self.m = m
          # rho is the rate at which the price of a derivative changes relative to a change in the risk-free rate of interest
          self.rho = rho
          self.optionType = optionType
          self.ctrlVar = ctrlVar
        except ValueError:
          print('Error passing Options parameters')
        if optionType != 'call' and optionType != 'put':
          raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
        if s01 < 0 or s02 < 0 or r < 0 or T <= 0 or K < 0 or sigma1 < 0 or sigma2 < 0:
          raise ValueError('Error: Negative inputs not allowed')

    # randoms = observation time in Mente Carlo
    def MonteCarloPrice(self):
        s0 = np.sqrt(self.s01 * self.s02)
        # n = # of baskets in Mean Basket Options
        n = 2
        m = self.m
        df = np.exp(-self.r * self.T)
        # SigmaBg Square
        sigsqT = (self.sigma1**2 + 2 *self.sigma1 * self.sigma2 * self.rho + self.sigma2**2) / (n**2)
        # mu multiply by T
        mu = (self.r - 0.5 * ((self.sigma1**2+self.sigma2**2) / n) + 0.5 * sigsqT)
        muT = mu * self.T
        drift1 = np.exp((self.r - 0.5 * self.sigma1 ** 2) * self.T)
        drift2 = np.exp((self.r - 0.5 * self.sigma2 ** 2) * self.T)

        # d1^ = d2^ + σ^√T = ln(S0/K) + (μ^+0.5σ^²)T / σ√T
        d1 = (np.log(s0 / self.K) + (muT + 0.5 * sigsqT * self.T)) / (np.sqrt(sigsqT) * np.sqrt(self.T))
        d2 = d1 - np.sqrt(sigsqT) * np.sqrt(self.T)
        N1, N2, N1_, N2_ = norm.cdf(d1), norm.cdf(d2), norm.cdf(-d1), norm.cdf(-d2)
        basket, arithPayoff, geoPayoffCall, geoPayoffPut = [0] * m, [0] * m, [0] * m, [0] * m

        for i in range(m):
            # fix the initial state
            initState = i * 2
            np.random.seed(initState)
            Z1 = np.random.normal(0, 1, 1)
            Z2 = self.rho * Z1 + np.sqrt(1-self.rho**2) * np.random.normal(0, 1, 1)
            ### si = s0e^(r-0.5σ²)dt + σ√Tξi continuous asset model ###
            S1 = self.s01 * drift1 * np.exp(self.sigma1 * np.sqrt(self.T) * Z1)
            S2 = self.s02 * drift2 * np.exp(self.sigma2 * np.sqrt(self.T) * Z2)
            
            # basket of two assets
            basket = (S1 + S2) * (1 / n)
            if self.optionType == "call":
                arithPayoff[i] = basket - self.K
                if (arithPayoff[i] > 0):
                  arithPayoff[i] = df * arithPayoff[i]
                else:
                  arithPayoff[i] = 0
            elif self.optionType == "put":
                arithPayoff[i] = self.K - basket
                if (arithPayoff[i] > 0):
                  arithPayoff[i] = df * arithPayoff[i]
                else:
                  arithPayoff[i] = 0

            ### Geometric mean
            geoMean = np.exp((1 / n) * (np.log(S1) + np.log(S2)))
            geoPayoffCall[i] = df * max(geoMean - self.K, 0)
            geoPayoffPut[i] = df * max(self.K - geoMean, 0)           
                
        ### without Control Variate ###
        if not self.ctrlVar:
            print('Mente Carlo method without control variate.')
            Pmean = float(np.mean(arithPayoff))
            Pstd = np.std(arithPayoff)
            lowerBound, upperBound = float(Pmean-1.96*Pstd / np.sqrt(m)), float(Pmean+1.96*Pstd / np.sqrt(m))
            confidentInterval = (lowerBound, upperBound)
            print('The {} basket options price is {} with {} confidence interval'.format(self.optionType, str(round(Pmean, 8)), confidentInterval))
            return Pmean, confidentInterval
        
        ### with Control variate ###
        callPriceMean, putPriceMean = np.mean(arithPayoff), np.mean(arithPayoff)
        if self.ctrlVar:
            print('Mente Carlo method with control variate.')
            geoCallPriceMean = np.mean(geoPayoffCall)
            geoPutPriceMean = np.mean(geoPayoffPut)
            arithPayoffMean = np.mean(arithPayoff)
            # cov(X,Y) = E[XY]−(EX)(EY)
            convXYCall = np.mean(np.multiply(arithPayoff, geoPayoffCall)) - (arithPayoffMean * geoCallPriceMean)
            convXYPut = np.mean(np.multiply(arithPayoff, geoPayoffPut)) - (arithPayoffMean * geoPutPriceMean)
            thetaCall = convXYCall / np.var(geoPayoffCall)
            thetaPut = convXYPut / np.var(geoPayoffPut)
            
            # # e^-rt(S0e^ρT N(d1) − KN(d2))
            if self.optionType == 'call':
                # closed-from formula for Geometric Mean Basket Call Option
                geoCall = df * (s0 * np.exp(muT) * N1 - self.K * N2)
                Z = arithPayoff + thetaCall * (geoCall - geoPayoffCall)
            elif self.optionType == 'put':
                # closed-from formula for Geometric Mean Basket Put Option
                geoPut = df * (self.K * N2_ - s0 * np.exp(muT) * N1_)
                Z = arithPayoff + thetaPut * (geoPut - geoPayoffPut)

            Zmean, Zstd = float(np.mean(Z)), np.std(Z)
            lowerBound, upperBound = float(Zmean-1.96 * Zstd / np.sqrt(m)), float(Zmean+1.96 * Zstd / np.sqrt(m))
            confidentInterval = (lowerBound, upperBound)
            print('The {} basket options price is {} with {} confidence interval.'.format(self.optionType, str(round(Zmean, 8)), confidentInterval))
            return Zmean, confidentInterval

if __name__ == '__main__':
    option = MonteCarloArithBasketOption(s01=100, s02=100, r=0.05, T=3, K=100, sigma1=0.1, sigma2=0.3, rho=0.5, optionType='call', m=100000, ctrlVar=False)
    option.MonteCarloPrice()
