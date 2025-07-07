# import numpy.testing as npt
# from pyscf import gto, scf
# from downfolding import Driver

# def _check_ducc(approx, expected):
#     # common setup
#     mol = gto.M(
#         atom=[["Be", (0.0, 0.0, 0.0)]],
#         basis="cc-pvdz",
#         charge=0,
#         spin=0,
#     )
#     mf = scf.RHF(mol)
#     mf.kernel(verbose=0)
#     driver = Driver.from_pyscf(mf, nfrozen=0)

#     # run and test
#     driver.run_ducc(n_act=5, approximation=approx, three_body=True, four_body=True)
#     for backend in ("openfermion",):
#         energy = driver.exact_diagonalize(backend=backend)
#         npt.assert_allclose(energy, expected,atol=1e-8, rtol=0,err_msg=f"DUCC {approx!r} {backend} energy mismatch")


# def test_004_ducc_a5():
#     _check_ducc("a5", -0.044927131205821)

# def test_004_ducc_a6():
#     _check_ducc("a6", -0.045684118792039)

# def test_004_ducc_a7():
#     _check_ducc("a7", -0.045708465026364)

# # test_004_ducc_a5()
# # test_004_ducc_a6()
# # test_004_ducc_a7()