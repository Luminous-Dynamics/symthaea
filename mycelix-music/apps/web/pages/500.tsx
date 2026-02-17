export default function ServerError() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex items-center justify-center p-6">
      <div className="bg-white/5 border border-white/10 rounded-xl p-8 text-center">
        <h1 className="text-4xl font-bold text-white mb-2">500</h1>
        <p className="text-gray-400">An unexpected error occurred. Please try again later.</p>
      </div>
    </div>
  );
}

