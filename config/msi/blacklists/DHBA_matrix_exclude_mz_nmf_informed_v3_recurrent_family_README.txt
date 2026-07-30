DHBA recurrent-family blacklist v3

Base:
  blacklists/DHBA_matrix_exclude_mz_nmf_informed_v2.txt

Output:
  blacklists/DHBA_matrix_exclude_mz_nmf_informed_v3_recurrent_family.txt

Purpose:
  Conservative v2 DHBA filtering left repeated DHBA-dominant families in several NMF factors.
  This v3 test additionally removes:
    - low-mass 169/170/171 DHBA-like family
    - recurrent high-mass 758/760/780/782/786/796/798 lipid/adduct-like family

Important:
  This is an exploratory cleaner blacklist, not final biology.
  The high-mass family could include real lipid biology, so compare v2 vs v3 before adopting.
