# Knowledge Base Index

## Papers

```dataview
TABLE title, year, claim_count, verified_claims, compiled_at
FROM "papers"
WHERE note_type = "paper"
SORT year DESC
```

## Concepts

```dataview
TABLE name, domain, claim_count
FROM "concepts"
WHERE note_type = "concept"
SORT name ASC
```
