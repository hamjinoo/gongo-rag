import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "DocLens Trace",
    template: "%s",
  },
  description:
    "답변에 쓰인 근거를 원문까지 추적하고 검색 과정을 검증하는 RAG 포트폴리오",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
