/*
  Retrieve canonical SMILES strings for Stage 1.
  Exclude the molecules that were tested on targets
  5-HT2A, Plasmodium falciparum, and  Staphylococcus aureus,
  as they will be analyzed in rediscovery studies in Stage 2.
*/

WITH target_ids AS (
  SELECT tid
  FROM target_dictionary
  WHERE
        pref_name LIKE 'Plasmodium falciparum%'
    AND pref_name LIKE 'Staphylococcus aureus%'
    AND pref_name IN (
      'Serotonin 2a (5-HT2a) receptor',
      '5-hydroxytryptamine receptor 2A',
      'Serotonin receptor 2a and 2c (5HT2A and 5HT2C)',
      'Serotonin receptor 2a and 2b (5HT2A and 5HT2B)',
      'Serotonin 2 receptors; 5-HT2a & 5-HT2c',
      'Dopamine D2 receptor and Serotonin 2a receptor (D2 and 5HT2a)'
    )
), assay_ids AS (
  SELECT assay_id
  FROM assays
  WHERE tid NOT IN target_ids
), molregnos AS (
  SELECT DISTINCT molregno
  FROM activities
  WHERE
    assay_id IN assay_ids
    AND standard_type IN ('IC50', 'MIC')
    AND standard_relation IN ('<', '<<', '<=', '=')
    AND standard_units = 'nM'
)

SELECT DISTINCT canonical_smiles
FROM compound_structures
WHERE molregno IN molregnos;