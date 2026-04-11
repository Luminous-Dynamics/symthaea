import { Request, Response } from 'express';
import { trustService } from '../services/trust.service';

export const trustController = {
  getSummary: async (req: Request, res: Response) => {
    const sender = (req.query.sender as string) || '';

    const summary = await trustService.getSummary(sender);

    res.json({
      status: 'success',
      data: { summary },
    });
  },

  clearCache: (_req: Request, res: Response) => {
    trustService.clearCache();
    res.json({ status: 'success', data: { message: 'Trust cache cleared' } });
  },

  health: (_req: Request, res: Response) => {
    const info = trustService.getProviderStatus();
    res.json({ status: 'success', data: info });
  },
};
