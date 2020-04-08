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
WHERE
  activities.assay_id IN (

    SELECT assay_id
    FROM assays
    WHERE
      tid IN (

        SELECT tid
        FROM target_dictionary
        WHERE
          -- Here we ignore not only the targets of interest but also the targets of
          -- akin target names.
          -- ??? Should we exclude only exact matches, i.e.
          --     target.dictionary.pref_name != 'Plasmodium falciparum'?
              pref_name NOT LIKE 'Plasmodium falciparum%'
          AND pref_name NOT LIKE 'Staphylococcus aureus%'
          AND pref_name NOT IN (
		    'Serotonin 2a (5-HT2a) receptor',
			'5-hydroxytryptamine receptor 2A',
			'Serotonin receptor 2a and 2c (5HT2A and 5HT2C)',
			'Serotonin receptor 2a and 2b (5HT2A and 5HT2B)',
			'Serotonin 2 receptors; 5-HT2a & 5-HT2c',
			'Dopamine D2 receptor and Serotonin 2a receptor (D2 and 5HT2a)'
		  )
      )

  )

  AND activities.standard_type IN ('IC50', 'MIC')
  AND activities.standard_relation IS NOT NULL;