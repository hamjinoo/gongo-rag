type BrandProps = {
  admin?: boolean;
};

export function Brand({ admin = false }: BrandProps) {
  return (
    <span className="brand" aria-label="DocLens Trace">
      <span>DocLens</span> <strong>Trace</strong>
      {admin ? <span className="admin-badge">관리자</span> : null}
    </span>
  );
}

