/*
  Retrieve canonical SMILES strings for Stage 1.
  Exclude the molecules that were tested on targets
  5-HT2A, Plasmodium falciparum, and  Staphylococcus aureus,
  as they will be analyzed in rediscovery studies in Stage 2.
*/

SELECT DISTINCT canonical_smiles
FROM compound_structures
  JOIN activities
    ON compound_structures.molregno = activities.molregno
  JOIN assays
    ON activities.assay_id = assays.assay_id
  JOIN target_dictionary
    ON assays.tid = target_dictionary.tid
WHERE
  -- Here we ignore not only the targets of interest but also the targets of
  -- akin target names.
  -- ??? Should we exclude only exact matches, i.e.
  --     target.dictionary.pref_name != 'Plasmodium falciparum'?
      target_dictionary.pref_name NOT LIKE 'Plasmodium falciparum%'
  AND target_dictionary.pref_name NOT LIKE 'Staphylococcus aureus%'
  AND target_dictionary.pref_name != 'Serotonin 2a (5-HT2a) receptor'
  AND target_dictionary.pref_name != '5-hydroxytryptamine receptor 2A';
