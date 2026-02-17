import { Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import { prisma } from '../utils/prisma';
import { asyncHandler } from '../middleware/errorHandler';
import { AppError } from '../middleware/errorHandler';

export class FolderController {
  getAll = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { accountId } = req.query;

    const where: any = { userId: req.userId! };
    if (accountId) where.emailAccountId = accountId as string;

    const folders = await prisma.folder.findMany({
      where,
      orderBy: { name: 'asc' },
    });

    res.json({ status: 'success', data: { folders } });
  });

  create = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { emailAccountId, name, path } = req.body;

    const folder = await prisma.folder.create({
      data: {
        userId: req.userId!,
        emailAccountId,
        name,
        path,
        type: 'CUSTOM',
      },
    });

    res.status(201).json({ status: 'success', data: { folder } });
  });

  update = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { name } = req.body;

    const folder = await prisma.folder.findFirst({
      where: { id: req.params.id, userId: req.userId! },
    });

    if (!folder) {
      throw new AppError('Folder not found', 404);
    }

    const updated = await prisma.folder.update({
      where: { id: req.params.id },
      data: { name },
    });

    res.json({ status: 'success', data: { folder: updated } });
  });

  delete = asyncHandler(async (req: AuthRequest, res: Response) => {
    const folder = await prisma.folder.findFirst({
      where: { id: req.params.id, userId: req.userId! },
    });

    if (!folder) {
      throw new AppError('Folder not found', 404);
    }

    if (folder.type !== 'CUSTOM') {
      throw new AppError('Cannot delete system folders', 400);
    }

    await prisma.folder.delete({
      where: { id: req.params.id },
    });

    res.json({ status: 'success', data: { message: 'Folder deleted' } });
  });
}

export const folderController = new FolderController();
