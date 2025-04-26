export default function ApiTestLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="container mx-auto">
      {children}
    </div>
  )
}
