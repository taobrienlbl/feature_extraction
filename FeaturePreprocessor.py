import shtns
import numpy as np

class FeaturePreprocessor:
    """ Pre-process environmental data associated with a weather feature. The main use is to center the feature in the dataset (and maybe rotate it). """

    def __init__(
            self,
            lat : np.ndarray,
            lon : np.ndarray,
            ntrunc : int = 85,
            gridtype = 'regular'):
        """ Initialize the feature pre-processor."""
        # Set transform parameters
        nlat = len(lat)
        nlon = len(lon)
        ntrunc = ntrunc 

        # store the input arguments
        self.lat = lat
        self.lon = lon
        self.ntrunc = ntrunc
        self.gridtype = gridtype

        # initialize the spherical harmonic transform
        self.transform = shtns.sht(ntrunc, ntrunc, 1, \
                            shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)

        # set the spherical harmonic grid
        if gridtype == "gaussian":
            self.transform.set_grid(nlat, nlon,
                    shtns.sht_quick_init | shtns.SHT_PHI_CONTIGUOUS, 1.e-10)
        elif gridtype == "regular":
            self.transform.set_grid(nlat, nlon,
                    shtns.sht_reg_dct | shtns.SHT_PHI_CONTIGUOUS, 1.e-10)

        # verify that the latitudes match
        transform_lat = np.rad2deg(np.arcsin(self.transform.cos_theta))[::-1]
        if not np.allclose(transform_lat, lat):
            print(f"Input latitudes: {lat}")
            print(f"shtns latitudes: {transform_lat}")
            raise RuntimeError("The latitudes of the transform do not match the input latitudes; check the gridtype.")

        # initialize the rotation operator
        self.rotation = shtns.rotation(self.transform.lmax, self.transform.mmax)

    def scalar_center_and_rotate(
            self,
            data2d : np.ndarray,
            clat : float,
            clon : float,
            angle : float = 0,
            inplace : bool = False,
            ):
        """ Rotate a 2d scalar lat/lon field so that it the point (clat, clon) is at the center of the field and is rotated by `angle` relative to the local East-West direction.

        Parameters
        ----------
        data2d : 2d array
            The data to be rotated.

        clat : float
            The latitude of the center point [degrees_north].

        clon : float
            The longitude of the center point [degrees_east].

        angle : float
            The angle to rotate the field by, relative to the local E-W axis [degrees].

        inplace : bool
            If True, the input data will be modified in place. If False, a copy of the data will be rotated and returned.
        
        Returns
        -------
        rotated_data2d : 2d array or None
            The rotated data if inplace=False, otherwise None.
        """

        # make sure that the data are array-like
        data2d = np.asarray(data2d)

        # make sure the data is a 2d array
        if len(data2d.shape) != 2:
            raise ValueError("data2d must be a 2d array.")

        # make sure longitude is in the range [0,360]
        if clon < 0:
            clon += 360
        if clon > 360 or clon < 0:
            raise ValueError(f"clon = {clon}, but clon must be in the range [-180,180] or [0,360].")

        # make the latitude in the range [-90,90]
        if clat > 90 or clat < -90:
            raise ValueError(f"clat = {clat}, but clat must be in the range [-90,90].")

        # make sure that the angle is in the range [0,360]
        if angle < 0 or angle > 360:
            raise ValueError(f"angle = {angle}, but angle must be in the range [0,360].")

        # set the first angle such that it rotates the data to lon=180 point
        zangle = 180 - clon

        # set the second angle to rotate the data to the pole
        yangle = 90-clat
        rotation_angle = angle

        # set the first rotation to rotate the data to the pole & rotate the data about the pole by `angle`
        self.rotation.set_angles_ZYZ(np.deg2rad(zangle),np.deg2rad(yangle),np.deg2rad(rotation_angle))

        # take the transform of the data
        data2d_sh = self.transform.analys(data2d.astype(np.float64))

        # rotate the data to the pole & rotate the data about the pole by `angle`
        data2d_rotated_sh = self.rotation.apply_real(data2d_sh)

        # set the second rotation to rotate the data to the equator
        self.rotation.set_angles_ZYZ(0,-np.pi/2,0)

        # rotate the data to the equator
        data2d_rotated_sh = self.rotation.apply_real(data2d_rotated_sh)

        # compute the inverse transform
        if inplace:
            # modify the input data in place
            data2d[:] = self.transform.synth(data2d_rotated_sh)
            return None
        else:
            # return the rotated data
            return self.transform.synth(data2d_rotated_sh)

        
        
