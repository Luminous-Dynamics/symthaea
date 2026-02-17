/**
 * Schemas Index
 *
 * Central export for all Zod validation schemas.
 */

// Common schemas
export {
  walletAddressSchema,
  uuidSchema,
  ipfsHashSchema,
  paginationSchema,
  sortSchema,
  searchSchema,
  dateRangeSchema,
  numericStringSchema,
  paymentModelSchema,
  transactionStatusSchema,
  genreSchema,
  safeStringSchema,
  titleSchema,
  descriptionSchema,
  urlSchema,
  idParamSchema,
  addressParamSchema,
} from './common.schema';

export type {
  WalletAddress,
  PaginationParams,
  SortParams,
  PaymentModel,
  TransactionStatus,
} from './common.schema';

// Song schemas
export {
  createSongSchema,
  updateSongSchema,
  songQuerySchema,
  songResponseSchema,
  batchSongIdsSchema,
  registerSongSchema,
  setClaimStreamSchema,
} from './song.schema';

export type {
  CreateSongInput,
  UpdateSongInput,
  SongQueryParams,
  SongResponse,
  BatchSongIds,
} from './song.schema';

// Play schemas
export {
  createPlaySchema,
  confirmPlaySchema,
  playQuerySchema,
  playResponseSchema,
  playWithSongResponseSchema,
  batchCreatePlaysSchema,
  playStatsQuerySchema,
} from './play.schema';

export type {
  CreatePlayInput,
  ConfirmPlayInput,
  PlayQueryParams,
  PlayResponse,
  PlayWithSongResponse,
  PlayStatsQuery,
} from './play.schema';
