# Data Retention and Privacy

## Customer data retention

Production customer records are retained for 7 years after contract end unless
a deletion request is received earlier under GDPR or CCPA.

## Backup retention

Database backups are kept for 35 days with point-in-time recovery enabled on
production clusters only.

## PII handling

Personally identifiable information must be encrypted at rest and masked in
non-production environments. Exporting PII requires legal approval.

## Right to erasure

Deletion requests must be completed within 30 days. Verification email is
sent to the requester when processing finishes.
