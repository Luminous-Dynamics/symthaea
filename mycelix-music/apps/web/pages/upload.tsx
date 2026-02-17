import { useState } from 'react';
import { usePrivy } from '@privy-io/react-auth';
import { ethers } from 'ethers';
import { Upload, Music, Check, ArrowRight, Image, AlertTriangle } from 'lucide-react';
import { EconomicStrategyWizard } from '../src/components/EconomicStrategyWizard';
import { EconomicStrategySDK } from '@/lib';
import type { EconomicConfig } from '@/lib';
import { mockSongs } from '../data/mockSongs';
import Navigation from '../components/Navigation';

export default function UploadPage() {
  const { authenticated, user, login } = usePrivy();
  const [currentStep, setCurrentStep] = useState(1);
  const [songFile, setSongFile] = useState<File | null>(null);
  const [coverFile, setCoverFile] = useState<File | null>(null);
  const [coverPreview, setCoverPreview] = useState<string | null>(null);
  const [songMetadata, setSongMetadata] = useState({
    title: '',
    artist: '',
    genre: '',
    description: '',
  });
  const [isUploading, setIsUploading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [duplicateWarning, setDuplicateWarning] = useState<string | null>(null);

  if (!authenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
        <Navigation />
        <div className="flex items-center justify-center min-h-screen pt-16">
          <div className="text-center">
            <Music className="w-16 h-16 text-purple-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-white mb-4">Connect Your Wallet</h2>
            <p className="text-gray-400 mb-8">You need to connect your wallet to upload music</p>
            <button
              onClick={login}
              className="px-8 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition"
            >
              Connect Wallet
            </button>
          </div>
        </div>
      </div>
    );
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && (file.type === 'audio/mpeg' || file.type === 'audio/flac')) {
      setSongFile(file);
      setCurrentStep(2);
    } else {
      alert('Please select an MP3 or FLAC file');
    }
  };

  const handleCoverSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      if (file.size > 5 * 1024 * 1024) {
        alert('Cover image must be less than 5MB');
        return;
      }
      setCoverFile(file);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setCoverPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      alert('Please select an image file (JPG, PNG, GIF)');
    }
  };

  const checkForDuplicates = (title: string, artist: string) => {
    // Check if song already exists in database (using mockSongs for demo)
    const duplicate = mockSongs.find(
      (song) =>
        song.title.toLowerCase() === title.toLowerCase() &&
        song.artist.toLowerCase() === artist.toLowerCase()
    );

    if (duplicate) {
      setDuplicateWarning(
        `⚠️ A song with this title and artist already exists (${duplicate.plays.toLocaleString()} plays). Are you sure this is a new upload?`
      );
    } else {
      setDuplicateWarning(null);
    }
  };

  const handleMetadataSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Check for duplicates
    checkForDuplicates(songMetadata.title, songMetadata.artist);

    // If there's a duplicate warning, require explicit confirmation
    if (duplicateWarning) {
      const confirmed = confirm(
        `${duplicateWarning}\n\nClick OK to continue anyway, or Cancel to go back and edit.`
      );
      if (!confirmed) {
        return;
      }
    }

    setCurrentStep(3);
  };

  const handleStrategyComplete = async (config: EconomicConfig) => {
    setIsUploading(true);

    try {
      // 1. Upload to IPFS (Web3.Storage)
      console.log('Uploading to IPFS...');
      const formData = new FormData();
      formData.append('file', songFile!);

      const ipfsResponse = await fetch('/api/upload-to-ipfs', {
        method: 'POST',
        body: formData,
      });
      const { ipfsHash } = await ipfsResponse.json();
      console.log('IPFS Hash:', ipfsHash);

      // 2. Register on-chain
      console.log('Registering on-chain...');
      const provider = new ethers.BrowserProvider((window as any).ethereum);
      const signer = await provider.getSigner();

      const sdk = new EconomicStrategySDK(
        provider,
        process.env.NEXT_PUBLIC_ROUTER_ADDRESS!,
        signer
      );

      const songId = `${songMetadata.artist}-${songMetadata.title}`;
      const txHash = await sdk.registerSong(songId, config);
      console.log('Transaction Hash:', txHash);

      // 3. Create DKG claim (epistemic)
      console.log('Creating DKG claim...');
      await fetch('/api/create-dkg-claim', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          songId,
          title: songMetadata.title,
          artist: songMetadata.artist,
          ipfsHash,
          artistAddress: await signer.getAddress(),
          epistemicTier: 'E3', // Cryptographically proven
          networkTier: 'N1', // Communal authority
          memoryTier: 'M3', // Permanent
        }),
      });

      setUploadComplete(true);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed: ' + (error as Error).message);
    } finally {
      setIsUploading(false);
    }
  };

  if (uploadComplete) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
            <Check className="w-12 h-12 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white mb-4">Song Uploaded Successfully! 🎉</h2>
          <p className="text-gray-400 mb-8">
            Your song &quot;{songMetadata.title}&quot; is now live on Mycelix Music
          </p>
          <div className="flex justify-center space-x-4">
            <button
              onClick={() => {
                setCurrentStep(1);
                setSongFile(null);
                setSongMetadata({ title: '', artist: '', genre: '', description: '' });
                setUploadComplete(false);
              }}
              className="px-6 py-3 border-2 border-purple-400 text-purple-400 hover:bg-purple-400/10 rounded-lg transition"
            >
              Upload Another
            </button>
            <a
              href="/discover"
              className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition"
            >
              View on Platform
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <Navigation />
      <div className="max-w-4xl mx-auto px-4 py-12 pt-24">
        {/* Progress Steps */}
        <div className="mb-12">
          <div className="flex items-center justify-between">
            {['Upload File', 'Song Details', 'Economic Strategy'].map((label, index) => (
              <div key={index} className="flex items-center">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                    index + 1 <= currentStep
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-700 text-gray-400'
                  }`}
                >
                  {index + 1}
                </div>
                <span
                  className={`ml-3 font-medium ${
                    index + 1 <= currentStep ? 'text-white' : 'text-gray-500'
                  }`}
                >
                  {label}
                </span>
                {index < 2 && <ArrowRight className="w-6 h-6 text-gray-600 mx-4" />}
              </div>
            ))}
          </div>
        </div>

        {/* Step 1: Upload File */}
        {currentStep === 1 && (
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8">
            <h2 className="text-2xl font-bold text-white mb-4">Upload Your Song</h2>
            <p className="text-gray-400 mb-6">
              Upload an MP3 or FLAC file. Maximum file size: 100MB
            </p>
            <label className="block cursor-pointer">
              <div className="border-2 border-dashed border-purple-400/50 rounded-xl p-12 text-center hover:border-purple-400 transition">
                <Upload className="w-16 h-16 text-purple-400 mx-auto mb-4" />
                <p className="text-white font-medium mb-2">Click to upload or drag and drop</p>
                <p className="text-gray-500 text-sm">MP3 or FLAC (max 100MB)</p>
              </div>
              <input
                type="file"
                accept="audio/mpeg,audio/flac"
                onChange={handleFileSelect}
                className="hidden"
              />
            </label>
            {songFile && (
              <div className="mt-4 p-4 bg-purple-600/20 border border-purple-400/30 rounded-lg">
                <p className="text-purple-300">
                  <strong>Selected:</strong> {songFile.name} ({(songFile.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              </div>
            )}
          </div>
        )}

        {/* Step 2: Song Details */}
        {currentStep === 2 && (
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8">
            <h2 className="text-2xl font-bold text-white mb-6">Song Details</h2>
            <form onSubmit={handleMetadataSubmit} className="space-y-4">
              <div>
                <label className="block text-gray-300 mb-2">Title</label>
                <input
                  type="text"
                  value={songMetadata.title}
                  onChange={(e) => setSongMetadata({ ...songMetadata, title: e.target.value })}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-purple-400 focus:outline-none"
                  placeholder="Enter song title"
                  required
                />
              </div>
              <div>
                <label className="block text-gray-300 mb-2">Artist Name</label>
                <input
                  type="text"
                  value={songMetadata.artist}
                  onChange={(e) => setSongMetadata({ ...songMetadata, artist: e.target.value })}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-purple-400 focus:outline-none"
                  placeholder="Your artist name"
                  required
                />
              </div>
              <div>
                <label className="block text-gray-300 mb-2">Genre</label>
                <select
                  value={songMetadata.genre}
                  onChange={(e) => setSongMetadata({ ...songMetadata, genre: e.target.value })}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-purple-400 focus:outline-none"
                  required
                >
                  <option value="">Select genre</option>
                  <option value="Electronic">Electronic</option>
                  <option value="Rock">Rock</option>
                  <option value="Hip-Hop">Hip-Hop</option>
                  <option value="Classical">Classical</option>
                  <option value="Jazz">Jazz</option>
                  <option value="Ambient">Ambient</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div>
                <label className="block text-gray-300 mb-2">Description</label>
                <textarea
                  value={songMetadata.description}
                  onChange={(e) => setSongMetadata({ ...songMetadata, description: e.target.value })}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:border-purple-400 focus:outline-none h-32"
                  placeholder="Tell listeners about your song..."
                />
              </div>

              {/* Album Cover Upload */}
              <div>
                <label className="block text-gray-300 mb-2">
                  Album Cover <span className="text-gray-500">(Optional but recommended)</span>
                </label>
                <div className="grid md:grid-cols-2 gap-4">
                  <label className="block cursor-pointer">
                    <div className="border-2 border-dashed border-purple-400/50 rounded-xl p-6 text-center hover:border-purple-400 transition">
                      <Image className="w-12 h-12 text-purple-400 mx-auto mb-2" />
                      <p className="text-white font-medium mb-1">Upload Cover Art</p>
                      <p className="text-gray-500 text-xs">JPG, PNG, or GIF (max 5MB)</p>
                      <p className="text-gray-500 text-xs mt-1">Recommended: 3000x3000px</p>
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleCoverSelect}
                      className="hidden"
                    />
                  </label>

                  {/* Cover Preview */}
                  {coverPreview && (
                    <div className="relative">
                      <img
                        src={coverPreview}
                        alt="Cover preview"
                        className="w-full aspect-square object-cover rounded-xl border-2 border-purple-400"
                      />
                      <button
                        type="button"
                        onClick={() => {
                          setCoverFile(null);
                          setCoverPreview(null);
                        }}
                        className="absolute top-2 right-2 px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm transition"
                      >
                        Remove
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Duplicate Warning */}
              {duplicateWarning && (
                <div className="p-4 bg-orange-500/20 border-2 border-orange-400 rounded-lg flex items-start space-x-3">
                  <AlertTriangle className="w-6 h-6 text-orange-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-orange-300 font-medium mb-1">Possible Duplicate Detected</p>
                    <p className="text-orange-200 text-sm">{duplicateWarning}</p>
                  </div>
                </div>
              )}

              <div className="flex space-x-4">
                <button
                  type="button"
                  onClick={() => setCurrentStep(1)}
                  className="px-6 py-3 border border-white/20 text-white rounded-lg hover:bg-white/5 transition"
                >
                  Back
                </button>
                <button
                  type="submit"
                  className="flex-1 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition"
                >
                  Next: Economic Strategy
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Step 3: Economic Strategy */}
        {currentStep === 3 && (
          <EconomicStrategyWizard
            songId={`${songMetadata.artist}-${songMetadata.title}`}
            onComplete={handleStrategyComplete}
            isLoading={isUploading}
          />
        )}
      </div>
    </div>
  );
}
