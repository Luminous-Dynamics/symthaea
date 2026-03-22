export default function NotFound() {
  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-6xl font-bold mb-4">404</h1>
        <h2 className="text-2xl mb-4">Page Not Found</h2>
        <p className="text-gray-400 mb-8">The page you're looking for doesn't exist.</p>
        <a 
          href="/"
          className="inline-block px-6 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg hover:opacity-90 transition"
        >
          Return Home
        </a>
      </div>
    </div>
  )
}