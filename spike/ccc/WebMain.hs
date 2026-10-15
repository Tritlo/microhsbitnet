module WebMain where

import WebApp (application)
import WebBrowser (runBrowser)

foreign export javascript "gpuMain" main :: IO ()

-- | Run the MicroHs application through the asynchronous browser bridge.
main :: IO ()
main = runBrowser application
