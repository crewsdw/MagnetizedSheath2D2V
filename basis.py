import numpy as np
import cupy as cp
import numpy.polynomial as poly
import scipy.special as sp

# Legendre-Gauss-Lobatto nodes and quadrature weights dictionaries
lgl_nodes = {
    1: [0],
    2: [-1, 1],
    3: [-1, 0, 1],
    4: [-1, -np.sqrt(1 / 5), np.sqrt(1 / 5), 1],
    5: [-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1],
    6: [-1, -np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21), -np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21),
        np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21), np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21), 1],
    7: [-1, -0.830223896278566929872, -0.468848793470714213803772,
        0, 0.468848793470714213803772, 0.830223896278566929872, 1],
    8: [-1, -0.8717401485096066153375, -0.5917001814331423021445,
        -0.2092992179024788687687, 0.2092992179024788687687,
        0.5917001814331423021445, 0.8717401485096066153375, 1],
    9: [-1, -0.8997579954114601573124, -0.6771862795107377534459,
        -0.3631174638261781587108, 0, 0.3631174638261781587108,
        0.6771862795107377534459, 0.8997579954114601573124, 1],
    10: [-1, -0.9195339081664588138289, -0.7387738651055050750031,
         -0.4779249498104444956612, -0.1652789576663870246262,
         0.1652789576663870246262, 0.4779249498104444956612,
         0.7387738651055050750031, 0.9195339081664588138289, 1],
    12: [-1, -0.9448992722228822234076, -0.8192793216440066783486,
         -0.6328761530318606776624, -0.3995309409653489322643,
         -0.1365529328549275548641, 0.1365529328549275548641,
         0.3995309409653489322643, 0.6328761530318606776624,
         0.8192793216440066783486, 0.9448992722228822234076, 1],
    15: [-1, -0.9652459265038385727959, -0.8850820442229762988254,
         -0.7635196899518152007041, -0.6062532054698457111235,
         -0.4206380547136724809219, -0.2153539553637942382257, 0,
         0.2153539553637942382257, 0.4206380547136724809219,
         0.6062532054698457111235, 0.7635196899518152007041,
         0.8850820442229762988254, 0.9652459265038385727959, 1],
    20: [-1, -0.9807437048939141719255, -0.9359344988126654357162,
         -0.8668779780899501413099, -0.7753682609520558704143,
         -0.6637764022903112898464, -0.5349928640318862616481,
         -0.3923531837139092993865, -0.239551705922986495182,
         -0.0805459372388218379759, 0.0805459372388218379759,
         0.239551705922986495182, 0.3923531837139092993865,
         0.5349928640318862616481, 0.6637764022903112898464,
         0.7753682609520558704143, 0.8668779780899501413099,
         0.9359344988126654357162, 0.9807437048939141719255, 1],
    25: [-1, -0.9877899449314937092718,
         -0.9592641382525344788599, -0.9149827707346225783232,
         -0.8556764658353165775238, -0.7823196592407167803992,
         -0.6961170488151343667604, -0.5984841472799932680976,
         -0.491024114818878382619, -0.3755014578592272332287,
         -0.2538130641688765801799, -0.127957059483106972709, 0,
         0.127957059483106972709, 0.2538130641688765801799,
         0.3755014578592272332287, 0.491024114818878382619,
         0.5984841472799932680976, 0.6961170488151343667604,
         0.782319659240716780399, 0.8556764658353165775238,
         0.9149827707346225783232, 0.9592641382525344788599,
         0.9877899449314937092718, 1],
    30: [-1, -0.9915739428405002933388, -0.9718466031662692416766,
         -0.9411047809510570823072, -0.899699218199276859553,
         -0.8480994871801981095514, -0.78689035723754708045,
         -0.716765398637085131634, -0.638519175807558407371,
         -0.5530382600950528523846, -0.4612911901682406852266,
         -0.364317500422448997756, -0.2632159437195737912671,
         -0.159132042625850467825, -0.053245110485486669363,
         0.05324511048548666936301, 0.159132042625850467825,
         0.2632159437195737912671, 0.364317500422448997756,
         0.461291190168240685227, 0.5530382600950528523846,
         0.6385191758075584073711, 0.716765398637085131634,
         0.78689035723754708045, 0.8480994871801981095514,
         0.8996992181992768595533, 0.9411047809510570823072,
         0.9718466031662692416766, 0.9915739428405002933388, 1]
}

lgl_weights = {
    1: [2],
    2: [1, 1],
    3: [1 / 3, 4 / 3, 1 / 3],
    4: [1 / 6, 5 / 6, 5 / 6, 1 / 6],
    5: [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10],
    6: [1 / 15, (14 - np.sqrt(7)) / 30, (14 + np.sqrt(7)) / 30,
        (14 + np.sqrt(7)) / 30, (14 - np.sqrt(7)) / 30, 1 / 15],
    7: [0.04761904761904761904762, 0.2768260473615659480107,
        0.4317453812098626234179, 0.487619047619047619048,
        0.4317453812098626234179, 0.2768260473615659480107,
        0.04761904761904761904762],
    8: [0.03571428571428571428571, 0.210704227143506039383,
        0.3411226924835043647642, 0.4124587946587038815671,
        0.4124587946587038815671, 0.3411226924835043647642,
        0.210704227143506039383, 0.03571428571428571428571],
    9: [0.02777777777777777777778, 0.1654953615608055250463,
        0.2745387125001617352807, 0.3464285109730463451151,
        0.3715192743764172335601, 0.3464285109730463451151,
        0.2745387125001617352807, 0.1654953615608055250463,
        0.02777777777777777777778],
    10: [0.02222222222222222222222, 0.1333059908510701111262,
         0.2248893420631264521195, 0.2920426836796837578756,
         0.3275397611838974566565, 0.3275397611838974566565,
         0.292042683679683757876, 0.224889342063126452119,
         0.133305990851070111126, 0.02222222222222222222222],
    12: [0.01515151515151515151515, 0.091684517413196130668,
         0.1579747055643701151647, 0.212508417761021145358,
         0.2512756031992012802932, 0.2714052409106961770003,
         0.2714052409106961770003, 0.251275603199201280293,
         0.212508417761021145358, 0.1579747055643701151647,
         0.0916845174131961306683, 0.01515151515151515151515],
    15: [0.009523809523809523809524, 0.0580298930286012490969,
         0.1016600703257180676037, 0.1405116998024281094605,
         0.1727896472536009490521, 0.196987235964613356093,
         0.2119735859268209201274, 0.217048116348815649515,
         0.2119735859268209201274, 0.1969872359646133560925,
         0.1727896472536009490521, 0.1405116998024281094605,
         0.1016600703257180676037, 0.0580298930286012490969, 0.009523809523809523809524],
    20: [0.005263157894736842105263, 0.03223712318848894149161,
         0.0571818021275668260048, 0.0806317639961196031448,
         0.101991499699450815684, 0.1207092276286747250994,
         0.1363004823587241844898, 0.1483615540709168258147,
         0.1565801026474754871582, 0.160743286387845749008,
         0.1607432863878457490077, 0.156580102647475487158,
         0.148361554070916825815, 0.1363004823587241844898,
         0.120709227628674725099, 0.1019914996994508156838,
         0.080631763996119603145, 0.057181802127566826005,
         0.032237123188488941492, 0.005263157894736842105263],
    25: [0.003333333333333333333333, 0.0204651689329743853085,
         0.0365047387942713720324, 0.051936228368491474643,
         0.0665137286753127846939, 0.0799987748362929818016,
         0.0921701399106204219127, 0.102828030347957830828,
         0.1117974662683208881562, 0.1189311794068118254094,
         0.1241120389379502906952, 0.1272549775383314470171,
         0.128308389298661928337, 0.1272549775383314470171,
         0.1241120389379502906952, 0.118931179406811825409,
         0.111797466268320888156, 0.1028280303479578308275,
         0.0921701399106204219127, 0.0799987748362929818016,
         0.06651372867531278469387, 0.051936228368491474643,
         0.036504738794271372032, 0.0204651689329743853085, 0.003333333333333333333333],
    30: [0.002298850574712643678161, 0.0141317993279053876407,
         0.0252831667405514022043, 0.036142094199408535315,
         0.0465906945331429274019, 0.0565111979230803833022,
         0.0657913363977900549441, 0.0743260033247182538341,
         0.0820185128334069147997, 0.0887817123197652101673,
         0.0945389751938608917811, 0.09922507100429983065768,
         0.1027869053072349894703, 0.1051841215964546498562,
         0.1063895587236679249476, 0.1063895587236679249476,
         0.1051841215964546498562, 0.1027869053072349894703,
         0.0992250710042998306577, 0.0945389751938608917811,
         0.088781712319765210167, 0.0820185128334069147997,
         0.0743260033247182538341, 0.0657913363977900549441,
         0.056511197923080383302, 0.0465906945331429274019,
         0.0361420941994085353147, 0.0252831667405514022043,
         0.0141317993279053876407, 0.002298850574712643678161]
}


class LGLBasis1D:
    """
    Class containing basis-related methods and properties for Lobatto-Gauss-Legendre points
    """

    def __init__(self, order):
        # parameters
        self.order = int(order)
        self.nodes, self.weights = (np.array(lgl_nodes.get(self.order, "nothing")),
                                    np.array(lgl_weights.get(self.order, "nothing")))
        self.device_weights = np.asarray(self.weights)

        # vandermonde matrix and inverse
        self.eigenvalues = self.set_eigenvalues()
        self.vandermonde = self.set_vandermonde()
        self.inv_vandermonde = self.set_inv_vandermonde()
        self.deriv_vandermonde = self.set_deriv_vandermonde()

        # DG matrices
        self.mass, self.inv_mass = None, None
        self.face_mass = None
        self.device_mass = None
        self.internal, self.numerical = None, None

        # Set matrices
        self.set_mass_matrix(), self.set_inv_mass_matrix()
        self.device_mass = cp.asarray(self.mass)
        self.set_internal_flux_matrix()
        self.set_numerical_flux_matrix()

        # Compute translation matrix
        self.translation_matrix = None
        self.set_translation_matrix()

        # compute derivative matrix
        self.derivative_matrix = None
        self.set_deriv_matrix()

        ### Matrices for elliptic class (not on device)
        # advection and stiffness matrices (inner product arrays)
        self.advection_matrix = None
        self.advection_matrix = self.set_advection_matrix()
        self.stiffness_matrix = self.advection_matrix.T

        # face mass matrix, first and last columns of identity
        self.face_mass = np.eye(self.order)[:, np.array([0, -1])]

        # Compute quadratic flux matrix
        # self.quadratic_flux_matrix = None
        # self.set_quadratic_flux_matrix()

    def set_eigenvalues(self):
        evs = np.array([(2.0 * s + 1) / 2.0 for s in range(self.order - 1)])
        return np.append(evs, (self.order - 1) / 2.0)

    def set_vandermonde(self):
        return np.array([[sp.legendre(s)(self.nodes[j])
                          for j in range(self.order)]
                         for s in range(self.order)])

    def set_deriv_vandermonde(self):
        return np.array([[sp.legendre(s).deriv()(self.nodes[j])
                          for j in range(self.order)]
                         for s in range(self.order)])

    def set_inv_vandermonde(self):
        return np.array([[self.weights[j] * self.eigenvalues[s] * sp.legendre(s)(self.nodes[j])
                          for j in range(self.order)]
                         for s in range(self.order)])

    def set_deriv_matrix(self):
        self.derivative_matrix = np.tensordot(self.inv_vandermonde, self.deriv_vandermonde, axes=([0], [0]))

    def set_mass_matrix(self):
        # Diagonal part
        approx_mass = np.diag(self.weights)

        # Off-diagonal part
        p = sp.legendre(self.order - 1)
        v = np.multiply(self.weights, p(self.nodes))
        a = -self.order * (self.order - 1) / (2.0 * (2.0 * self.order - 1))
        # calculate mass matrix
        self.mass = approx_mass + a * np.outer(v, v)

    def set_inv_mass_matrix(self):
        # Diagonal part
        approx_inv = np.diag(np.divide(1.0, self.weights))

        # Off-diagonal part
        p = sp.legendre(self.order - 1)
        v = p(self.nodes)
        b = self.order / 2
        # calculate inverse mass matrix
        self.inv_mass = approx_inv + b * np.outer(v, v)

    def set_internal_flux_matrix(self):
        # Compute internal flux array
        up = np.zeros((self.order, self.order))
        for i in range(self.order):
            for j in range(self.order):
                up[i, j] = self.weights[j] * sum(
                    (2 * s + 1) / 2 * sp.legendre(s)(self.nodes[i]) *
                    sp.legendre(s).deriv()(self.nodes[j]) for s in range(self.order))

        # Clear machine errors
        up[np.abs(up) < 1.0e-10] = 0

        self.internal = cp.asarray(up)

    def set_numerical_flux_matrix(self):
        self.numerical = cp.asarray(self.inv_mass[:, np.array([0, -1])])

    def set_translation_matrix(self):
        """ Create the translation matrix for velocity DG method """
        local_order = self.order
        gl_nodes, gl_weights = poly.legendre.leggauss(local_order)
        # Evaluate Legendre polynomials at finer grid
        ps = np.array([sp.legendre(s)(gl_nodes) for s in range(self.order)])
        # Interpolation polynomials at fine points
        ell = np.tensordot(self.inv_vandermonde, ps, axes=([0], [0]))
        # Compute the matrix elements
        translation_mass = np.array([[
            sum(gl_weights[s] * gl_nodes[s] * ell[i, s] * ell[j, s] for s in range(local_order))
            for j in range(self.order)]
            for i in range(self.order)])
        # Multiply by inverse mass matrix
        self.translation_matrix = np.matmul(self.inv_mass, translation_mass) + 0j

    def set_advection_matrix(self):
        adv = np.zeros((self.order, self.order))
        # Fill matrix
        for i in range(self.order):
            for j in range(self.order):
                adv[i, j] = self.weights[i] * self.weights[j] * sum(
                    self.eigenvalues[s] * sp.legendre(s)(self.nodes[i]) *
                    sp.legendre(s).deriv()(self.nodes[j]) for s in range(self.order))

        # Clean machine error
        adv[np.abs(adv) < 1.0e-15] = 0

        return adv
    