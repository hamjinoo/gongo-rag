import type { Metadata } from "next";
import { VerifyScreen } from "../../components/VerifyScreen";
import { getCitation } from "../../lib/mock-trace";

type VerifyPageProps = {
  params: Promise<{ citationId: string }>;
};

export async function generateMetadata({
  params,
}: VerifyPageProps): Promise<Metadata> {
  const { citationId } = await params;
  const citation = getCitation(citationId);

  return {
    title: `근거 ${citation.id} 검증 | DocLens Trace`,
    description: citation.claim,
  };
}

export default async function CitationVerifyPage({ params }: VerifyPageProps) {
  const { citationId } = await params;
  return <VerifyScreen citation={getCitation(citationId)} />;
}

