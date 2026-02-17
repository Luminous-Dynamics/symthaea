import { Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import { emailService } from '../services/email.service';
import { prisma } from '../utils/prisma';
import { asyncHandler } from '../middleware/errorHandler';
import { AppError } from '../middleware/errorHandler';

export class EmailController {
  getAll = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { accountId, folderId, page = '1', limit = '50', search } = req.query;

    const where: any = { userId: req.userId! };

    if (accountId) where.emailAccountId = accountId as string;
    if (folderId) where.folderId = folderId as string;
    if (search) {
      where.OR = [
        { subject: { contains: search as string, mode: 'insensitive' } },
        { bodyText: { contains: search as string, mode: 'insensitive' } },
      ];
    }

    const [emails, total] = await Promise.all([
      prisma.email.findMany({
        where,
        orderBy: { date: 'desc' },
        take: parseInt(limit as string),
        skip: (parseInt(page as string) - 1) * parseInt(limit as string),
        include: {
          folder: { select: { name: true, type: true } },
          attachments: { select: { id: true, filename: true, contentType: true, size: true } },
        },
      }),
      prisma.email.count({ where }),
    ]);

    res.json({
      status: 'success',
      data: {
        emails,
        pagination: {
          page: parseInt(page as string),
          limit: parseInt(limit as string),
          total,
          pages: Math.ceil(total / parseInt(limit as string)),
        },
      },
    });
  });

  getOne = asyncHandler(async (req: AuthRequest, res: Response) => {
    const email = await prisma.email.findFirst({
      where: { id: req.params.id, userId: req.userId! },
      include: {
        folder: true,
        attachments: true,
      },
    });

    if (!email) {
      throw new AppError('Email not found', 404);
    }

    // Mark as read
    if (!email.isRead) {
      await prisma.email.update({
        where: { id: email.id },
        data: { isRead: true },
      });
    }

    res.json({ status: 'success', data: { email } });
  });

  send = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { accountId, to, subject, body, cc, bcc, attachments } = req.body;

    const info = await emailService.sendEmail(
      req.userId!,
      accountId,
      to,
      subject,
      body,
      cc,
      bcc,
      attachments
    );

    res.status(201).json({ status: 'success', data: { messageId: info.messageId } });
  });

  markRead = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { isRead } = req.body;

    const email = await prisma.email.findFirst({
      where: { id: req.params.id, userId: req.userId! },
    });

    if (!email) {
      throw new AppError('Email not found', 404);
    }

    const updated = await prisma.email.update({
      where: { id: req.params.id },
      data: { isRead },
    });

    res.json({ status: 'success', data: { email: updated } });
  });

  markStarred = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { isStarred } = req.body;

    const email = await prisma.email.findFirst({
      where: { id: req.params.id, userId: req.userId! },
    });

    if (!email) {
      throw new AppError('Email not found', 404);
    }

    const updated = await prisma.email.update({
      where: { id: req.params.id },
      data: { isStarred },
    });

    res.json({ status: 'success', data: { email: updated } });
  });

  delete = asyncHandler(async (req: AuthRequest, res: Response) => {
    const email = await prisma.email.findFirst({
      where: { id: req.params.id, userId: req.userId! },
    });

    if (!email) {
      throw new AppError('Email not found', 404);
    }

    await prisma.email.delete({
      where: { id: req.params.id },
    });

    res.json({ status: 'success', data: { message: 'Email deleted' } });
  });

  sync = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { accountId, folderPath = 'INBOX' } = req.body;

    const result = await emailService.syncFolder(req.userId!, accountId, folderPath);

    res.json({ status: 'success', data: result });
  });
}

export const emailController = new EmailController();
