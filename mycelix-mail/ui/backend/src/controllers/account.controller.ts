import { Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import { accountService } from '../services/account.service';
import { asyncHandler } from '../middleware/errorHandler';

export class AccountController {
  create = asyncHandler(async (req: AuthRequest, res: Response) => {
    const account = await accountService.createAccount(req.userId!, req.body);
    res.status(201).json({ status: 'success', data: { account } });
  });

  getAll = asyncHandler(async (req: AuthRequest, res: Response) => {
    const accounts = await accountService.getAccounts(req.userId!);
    res.json({ status: 'success', data: { accounts } });
  });

  getOne = asyncHandler(async (req: AuthRequest, res: Response) => {
    const account = await accountService.getAccount(req.userId!, req.params.id);
    res.json({ status: 'success', data: { account } });
  });

  update = asyncHandler(async (req: AuthRequest, res: Response) => {
    const account = await accountService.updateAccount(req.userId!, req.params.id, req.body);
    res.json({ status: 'success', data: { account } });
  });

  delete = asyncHandler(async (req: AuthRequest, res: Response) => {
    const result = await accountService.deleteAccount(req.userId!, req.params.id);
    res.json({ status: 'success', data: result });
  });

  setDefault = asyncHandler(async (req: AuthRequest, res: Response) => {
    const account = await accountService.setDefaultAccount(req.userId!, req.params.id);
    res.json({ status: 'success', data: { account } });
  });
}

export const accountController = new AccountController();
