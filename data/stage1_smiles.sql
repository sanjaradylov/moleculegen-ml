SELECT cs.canonical_smiles
FROM compound_structures cs
    JOIN target_dictionary td ON td.tid = a.tid
    JOIN assays a ON a.assay_id = act.assay_id
    JOIN activities act ON act.molregno = cs.molregno
        AND td.organism NOT LIKE 'Plasmodium falciparum%'
        AND td.organism NOT LIKE 'Staphylococcus aureus%';
